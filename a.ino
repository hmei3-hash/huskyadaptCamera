// Ultrasonic presence detector for ESP32 Arduino.
//
// Design goals:
// - Measure distance without blocking the main loop.
// - Smooth noisy ultrasonic readings with a small median filter.
// - Trigger once when someone enters, then stay quiet until they fully leave.
// - Keep the code simple enough to debug from the Serial Monitor.

// === Pins ===
#define TRIG_PIN        18          // Ultrasonic sensor trigger pin
#define ECHO_PIN        5           // Ultrasonic sensor echo pin
#define TRIGGER_PIN     23          // Output pin to the external trigger device
#define BUTTON_PIN      19          // Mode button, wired to ground with INPUT_PULLUP

// === Distance behavior ===
#define MAX_DISTANCE_CM   400       // Ignore anything farther than this
#define THRESHOLD_CM      100       // "Occupied" begins closer than this distance
#define HYSTERESIS_CM     10        // Must clear past threshold + this amount to re-arm

// === Timing ===
#define PULSE_MS          1000      // Output trigger pulse length in queue/line modes
#define PING_INTERVAL_MS  40        // Time between ultrasonic pings
#define ECHO_TIMEOUT_US   25000     // Max echo wait before treating reading as no object
#define DEBOUNCE_MS       50        // Button debounce time
#define DEBUG_PRINT_MS    200       // Serial debug print interval

// === Sensor math ===
#define SOUND_SPEED_CM_US 0.0343f   // Speed of sound in cm per microsecond

// === Filtering and state ===
#define FILTER_SIZE       5         // Median filter sample count
#define CLEAR_READINGS    3         // Queue mode readings required before re-arming
#define LINE_CLEAR_EXTRA  2         // Line mode adds extra clear readings before re-arming

// === Modes ===
// QUEUE:
//   Trigger once when something enters the zone. Do not retrigger until clear.
//
// STRICT:
//   Keep the trigger output high while something is inside the zone.
//
// LINE:
//   Like queue mode, but requires more clear readings before re-arming.
//   This is more tolerant of slow or uneven movement.
enum Mode {
  MODE_QUEUE,
  MODE_STRICT,
  MODE_LINE,
  MODE_COUNT
};

// === Interrupt-shared echo state ===
// These are written by the interrupt and read by loop code, so they must be volatile.
volatile unsigned long echoRiseUs = 0;
volatile unsigned long echoPulseUs = 0;
volatile bool echoComplete = false;

// === Current mode ===
Mode currentMode = MODE_QUEUE;

// === Ping state ===
// These let the sketch start a ping, keep looping, and later handle the echo.
unsigned long lastPingMs = 0;
unsigned long pingStartUs = 0;
bool waitingForEcho = false;

// === Distance readings ===
unsigned long rawDistanceCm = MAX_DISTANCE_CM;
unsigned long filteredDistanceCm = MAX_DISTANCE_CM;
bool newDistanceAvailable = false;

// === Median filter buffer ===
unsigned long filterBuffer[FILTER_SIZE];
uint8_t filterIndex = 0;
uint8_t filterCount = 0;

// === Presence state ===
// occupied means the area is currently armed/claimed by a person or object.
// clearCount prevents one noisy far reading from re-arming the detector.
bool occupied = false;
uint8_t clearCount = 0;

// === Trigger output state ===
bool triggerActive = false;
unsigned long triggerStartMs = 0;

// === Button debounce state ===
bool buttonStableState = HIGH;
bool lastButtonReading = HIGH;
unsigned long lastButtonChangeMs = 0;

// === Debug timing ===
unsigned long lastDebugMs = 0;

// Function prototypes make this compatible with Arduino IDE's simple build flow.
void checkButton();
void advanceMode();
void updateSensor();
void startPing();
unsigned long pulseToDistanceCm(unsigned long pulseUs);
unsigned long getMedianDistance(unsigned long distanceCm);
void sortSmallArray(unsigned long values[], uint8_t count);
void updatePresence(unsigned long distanceCm);
uint8_t getRequiredClearReadings();
void updateTrigger();
void startTrigger();
void stopTrigger();
void printDebug();
const char *getModeName(Mode mode);

void IRAM_ATTR echoISR() {
  // Rising edge: echo pulse has started.
  if (digitalRead(ECHO_PIN) == HIGH) {
    echoRiseUs = micros();
  } else {
    // Falling edge: echo pulse has ended.
    echoPulseUs = micros() - echoRiseUs;
    echoComplete = true;
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(TRIGGER_PIN, LOW);

  // Start the filter with "far away" values so startup does not false-trigger.
  for (uint8_t i = 0; i < FILTER_SIZE; i++) {
    filterBuffer[i] = MAX_DISTANCE_CM;
  }

  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoISR, CHANGE);

  Serial.println("Presence detector ready");
  Serial.println("Mode: QUEUE");
}

void loop() {
  checkButton();
  updateSensor();

  // Only update the presence state when a fresh distance reading is ready.
  if (newDistanceAvailable) {
    newDistanceAvailable = false;
    filteredDistanceCm = getMedianDistance(rawDistanceCm);
    updatePresence(filteredDistanceCm);
  }

  updateTrigger();
  printDebug();
}

void checkButton() {
  bool reading = digitalRead(BUTTON_PIN);
  unsigned long now = millis();

  // Any change restarts the debounce timer.
  if (reading != lastButtonReading) {
    lastButtonChangeMs = now;
    lastButtonReading = reading;
  }

  // Accept the new button state only after it has stayed stable.
  if ((now - lastButtonChangeMs) >= DEBOUNCE_MS && reading != buttonStableState) {
    buttonStableState = reading;

    // INPUT_PULLUP means LOW is pressed.
    if (buttonStableState == LOW) {
      advanceMode();
    }
  }
}

void advanceMode() {
  currentMode = (Mode)((currentMode + 1) % MODE_COUNT);

  // Reset state when switching modes so the new mode starts cleanly.
  occupied = false;
  clearCount = 0;
  stopTrigger();

  Serial.print("Mode: ");
  Serial.println(getModeName(currentMode));
}

void updateSensor() {
  unsigned long nowMs = millis();
  unsigned long nowUs = micros();

  // Start a new ping at a fixed interval, as long as the previous ping is done.
  if (!waitingForEcho && (nowMs - lastPingMs) >= PING_INTERVAL_MS) {
    noInterrupts();
    echoComplete = false;
    interrupts();

    startPing();
    lastPingMs = nowMs;
    pingStartUs = nowUs;
    waitingForEcho = true;
  }

  // If the interrupt captured a complete echo pulse, convert it to distance.
  if (waitingForEcho && echoComplete) {
    noInterrupts();
    unsigned long pulseUs = echoPulseUs;
    echoComplete = false;
    interrupts();

    waitingForEcho = false;
    rawDistanceCm = pulseToDistanceCm(pulseUs);
    newDistanceAvailable = true;
  }

  // If no echo arrives in time, treat it as "far away."
  if (waitingForEcho && (micros() - pingStartUs) > ECHO_TIMEOUT_US) {
    noInterrupts();
    echoComplete = false;
    interrupts();

    waitingForEcho = false;
    rawDistanceCm = MAX_DISTANCE_CM;
    newDistanceAvailable = true;
  }
}

void startPing() {
  // HC-SR04-style sensors want a short 10 microsecond trigger pulse.
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
}

unsigned long pulseToDistanceCm(unsigned long pulseUs) {
  // The sound travels to the object and back, so divide by 2.
  unsigned long distance = (unsigned long)((pulseUs * SOUND_SPEED_CM_US) / 2.0f);

  if (distance == 0 || distance > MAX_DISTANCE_CM) {
    return MAX_DISTANCE_CM;
  }

  return distance;
}

unsigned long getMedianDistance(unsigned long distanceCm) {
  filterBuffer[filterIndex] = distanceCm;
  filterIndex = (filterIndex + 1) % FILTER_SIZE;

  if (filterCount < FILTER_SIZE) {
    filterCount++;
  }

  // Copy the active samples, sort the copy, and return the middle value.
  unsigned long sorted[FILTER_SIZE];

  for (uint8_t i = 0; i < filterCount; i++) {
    sorted[i] = filterBuffer[i];
  }

  sortSmallArray(sorted, filterCount);
  return sorted[filterCount / 2];
}

void sortSmallArray(unsigned long values[], uint8_t count) {
  // Tiny manual sort. This avoids STL and keeps the sketch Arduino-friendly.
  for (uint8_t i = 0; i < count; i++) {
    for (uint8_t j = i + 1; j < count; j++) {
      if (values[j] < values[i]) {
        unsigned long temp = values[i];
        values[i] = values[j];
        values[j] = temp;
      }
    }
  }
}

void updatePresence(unsigned long distanceCm) {
  bool insideZone = distanceCm < THRESHOLD_CM;
  bool fullyClear = distanceCm > (THRESHOLD_CM + HYSTERESIS_CM);

  if (currentMode == MODE_STRICT) {
    // Strict mode acts like a live presence output with hysteresis.
    if (!occupied && insideZone) {
      occupied = true;
    } else if (occupied && fullyClear) {
      occupied = false;
    }

    return;
  }

  // Queue/line modes trigger once on entry.
  if (!occupied && insideZone) {
    occupied = true;
    clearCount = 0;
    startTrigger();
    return;
  }

  // Once occupied, require several fully-clear readings before re-arming.
  if (occupied && fullyClear) {
    clearCount++;

    if (clearCount >= getRequiredClearReadings()) {
      occupied = false;
      clearCount = 0;
    }
  } else if (occupied && insideZone) {
    // If the person/object is clearly still inside, cancel the clear attempt.
    clearCount = 0;
  }
}

uint8_t getRequiredClearReadings() {
  if (currentMode == MODE_LINE) {
    return CLEAR_READINGS + LINE_CLEAR_EXTRA;
  }

  return CLEAR_READINGS;
}

void updateTrigger() {
  if (currentMode == MODE_STRICT) {
    // In strict mode, the output mirrors the occupied state.
    digitalWrite(TRIGGER_PIN, occupied ? HIGH : LOW);
    triggerActive = occupied;
    return;
  }

  // In queue/line modes, turn off the pulse after PULSE_MS.
  if (triggerActive && (millis() - triggerStartMs) >= PULSE_MS) {
    stopTrigger();
  }
}

void startTrigger() {
  digitalWrite(TRIGGER_PIN, HIGH);
  triggerActive = true;
  triggerStartMs = millis();
}

void stopTrigger() {
  digitalWrite(TRIGGER_PIN, LOW);
  triggerActive = false;
}

void printDebug() {
  unsigned long now = millis();

  if ((now - lastDebugMs) < DEBUG_PRINT_MS) {
    return;
  }

  lastDebugMs = now;

  Serial.print("mode=");
  Serial.print(getModeName(currentMode));
  Serial.print(" raw=");
  Serial.print(rawDistanceCm);
  Serial.print("cm filtered=");
  Serial.print(filteredDistanceCm);
  Serial.print("cm occupied=");
  Serial.print(occupied ? "yes" : "no");
  Serial.print(" clearCount=");
  Serial.print(clearCount);
  Serial.print(" trigger=");
  Serial.println(triggerActive ? "on" : "off");
}

const char *getModeName(Mode mode) {
  switch (mode) {
    case MODE_QUEUE:
      return "QUEUE";
    case MODE_STRICT:
      return "STRICT";
    case MODE_LINE:
      return "LINE";
    default:
      return "UNKNOWN";
  }
}
