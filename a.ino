#define TRIG_PIN        18
#define ECHO_PIN        5
#define TRIGGER_PIN     23
#define BUTTON_PIN      19

#define MAX_DISTANCE_CM   400
#define THRESHOLD_CM      100
#define PULSE_MS          1000      // 1 second trigger pulse
#define SOUND_SPEED_CM_US 0.0343f

// === Consensus ===
#define HISTORY_SIZE      20
#define SIMILARITY_CM     10
#define MIN_AGREEMENT     2

// === State confirmation (use a few readings to confirm in/out state, noise-resistant) ===
#define STATE_CONFIRM_N   3         // Require N consecutive readings to change inside/outside state
#define HYSTERESIS_CM     10        // Hysteresis band to prevent flickering at the threshold edge

// === Timing ===
#define PING_INTERVAL_MS  40
#define ECHO_TIMEOUT_US   25000
#define DEBUG_PRINT_MS    200
#define DEBOUNCE_MS       50

// === Modes ===
enum Mode { MODE_QUEUE, MODE_STRICT, MODE_COUNT };
volatile uint8_t currentMode = MODE_QUEUE;

// === ISR shared state ===
volatile unsigned long echoStartUs = 0;
volatile unsigned long echoEndUs = 0;
volatile bool echoDone = false;

// === Buffer & timing ===
unsigned long history[HISTORY_SIZE];
uint8_t historyIndex = 0;
bool historyFull = false;

unsigned long lastPingMs = 0;
unsigned long pingStartUs = 0;
bool waitingForEcho = false;

unsigned long triggerHighUntilMs = 0;
bool pinIsHigh = false;

unsigned long lastPrintMs = 0;
unsigned long lastDist = 0;
uint8_t lastAgreement = 0;

// === Inside/outside state ===
bool wasInside = false;             // Whether currently within the threshold
uint8_t insideCount = 0;            // Consecutive readings judged as inside
uint8_t outsideCount = 0;           // Consecutive readings judged as outside

// === Button state ===
bool lastButtonState = HIGH;
bool buttonStableState = HIGH;
unsigned long lastDebounceMs = 0;

void IRAM_ATTR echoISR() {
  if (digitalRead(ECHO_PIN) == HIGH) {
    echoStartUs = micros();         // Rising edge: echo pulse started
  } else {
    echoEndUs = micros();           // Falling edge: echo pulse ended
    echoDone = true;
  }
}

uint8_t countSimilar(unsigned long currentDist) {
  uint8_t count = 0;
  uint8_t total = historyFull ? HISTORY_SIZE : historyIndex;
  for (uint8_t i = 0; i < total; i++) {
    if (history[i] == 0) continue;
    long diff = (long)history[i] - (long)currentDist;
    if (diff < 0) diff = -diff;
    if (diff <= SIMILARITY_CM) count++;
  }
  return count;
}

void triggerPing() {
  // Send a 10us pulse on TRIG
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  echoDone = false;
  pingStartUs = micros();
  waitingForEcho = true;
}

void fireTrigger() {
  digitalWrite(TRIGGER_PIN, HIGH);
  pinIsHigh = true;
  triggerHighUntilMs = millis() + PULSE_MS;
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(TRIGGER_PIN, LOW);
  for (uint8_t i = 0; i < HISTORY_SIZE; i++) history[i] = 0;
  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoISR, CHANGE);
  lastPingMs = millis();
  Serial.println("Ready - Mode: QUEUE");
}

void processReading(unsigned long dist) {
  history[historyIndex] = dist;
  historyIndex = (historyIndex + 1) % HISTORY_SIZE;
  if (historyIndex == 0) historyFull = true;

  lastDist = dist;
  lastAgreement = (dist != 0) ? countSimilar(dist) : 0;

  // Invalid reading: don't change state (person may have left)
  if (dist == 0) {
    return;
  }

  // Judge inside/outside using hysteresis:
  // entering requires < THRESHOLD, leaving requires > THRESHOLD + hysteresis,
  // which avoids flickering at the edge
  bool readingInside  = (dist < THRESHOLD_CM);
  bool readingOutside = (dist > THRESHOLD_CM + HYSTERESIS_CM);

  // Reading only counts toward state if consensus is high enough
  bool reliable = (lastAgreement >= MIN_AGREEMENT);

  if (readingInside && reliable) {
    insideCount++;
    outsideCount = 0;
    // Confirmed entry, and previously outside -> crossing edge, fire trigger
    if (insideCount >= STATE_CONFIRM_N && !wasInside) {
      wasInside = true;
      fireTrigger();          // Fire only on the outside->inside transition
    }
  } else if (readingOutside) {
    outsideCount++;
    insideCount = 0;
    // Confirmed exit -> reset state so the next entry can trigger again
    if (outsideCount >= STATE_CONFIRM_N && wasInside) {
      wasInside = false;
    }
  }
  // Between THRESHOLD and THRESHOLD+HYSTERESIS: hold current state (hysteresis band)

  // STRICT mode: overrides the above; fires every time while inside if consensus is high enough
  if (currentMode == MODE_STRICT && readingInside && reliable) {
    fireTrigger();
  }
}

void checkButton() {
  bool reading = digitalRead(BUTTON_PIN);
  if (reading != lastButtonState) {
    lastDebounceMs = millis();   // State changed, reset debounce timer
  }
  if ((millis() - lastDebounceMs) > DEBOUNCE_MS) {
    if (reading != buttonStableState) {
      buttonStableState = reading;
      // On press (HIGH->LOW because of PULLUP), switch mode
      if (buttonStableState == LOW) {
        currentMode = (currentMode + 1) % MODE_COUNT;
        wasInside = false;       // Reset state when switching mode
        insideCount = 0;
        outsideCount = 0;
        Serial.print("Mode switched to: ");
        Serial.println(currentMode == MODE_QUEUE ? "QUEUE" : "STRICT");
      }
    }
  }
  lastButtonState = reading;
}

void loop() {
  unsigned long now = millis();

  checkButton();

  // Task 1: kick off a new ping every PING_INTERVAL_MS
  if (!waitingForEcho && (now - lastPingMs >= PING_INTERVAL_MS)) {
    lastPingMs = now;
    triggerPing();
  }

  // Task 2: handle finished echo
  if (waitingForEcho && echoDone) {
    waitingForEcho = false;
    unsigned long duration = echoEndUs - echoStartUs;
    unsigned long dist = (unsigned long)((duration * SOUND_SPEED_CM_US) / 2.0f);
    if (dist < 2 || dist > MAX_DISTANCE_CM) dist = 0;   // mark invalid
    processReading(dist);
  }

  // Task 3: timeout check (no echo received)
  if (waitingForEcho && (micros() - pingStartUs > ECHO_TIMEOUT_US)) {
    waitingForEcho = false;
    processReading(0);   // 0 = no reading
  }

  // Task 4: non-blocking pulse drop
  if (pinIsHigh && (long)(now - triggerHighUntilMs) >= 0) {
    digitalWrite(TRIGGER_PIN, LOW);
    pinIsHigh = false;
  }

  // Task 5: debug print (rate-limited to avoid blocking the main loop)
  if (now - lastPrintMs >= DEBUG_PRINT_MS) {
    lastPrintMs = now;
    Serial.print(currentMode == MODE_QUEUE ? "[Q] " : "[S] ");
    Serial.print("dist=");
    Serial.print(lastDist);
    Serial.print("cm inside=");
    Serial.print(wasInside ? "Y" : "N");
    Serial.print(" agree=");
    Serial.print(lastAgreement);
    Serial.println();
  }
}
