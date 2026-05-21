#include <Wire.h>

// ============================================================
//  WALKER PROXIMITY SENSOR
//  - Ultrasonic (HC-SR04) decides WHEN to alert
//  - MPU6050 (IMU)        decides WHICH mode we are in
//
//  WALKING mode  : moving walker. Alert ONCE per new obstacle.
//  QUEUING mode  : stopped in line. Alert ONCE when first too
//                  close, then stay quiet through normal shuffle.
// ============================================================

// === Pins (ESP32) ===
#define TRIG_PIN        4          // ultrasonic TRIG (was 18)
#define ECHO_PIN        5          // ultrasonic ECHO
#define TRIGGER_PIN     23         // alert output (buzzer/vibrator)
// I2C for MPU6050:  SDA = 21, SCL = 18
#define I2C_SDA         21
#define I2C_SCL         18
#define MPU_ADDR        0x68       // MPU6050 default I2C address

// === Distance thresholds ===
#define MAX_DISTANCE_CM   400
#define THRESHOLD_CM      100      // "too close" line
#define HYSTERESIS_CM     10       // walking re-arm band  (-> 110cm)
#define QUEUE_RELEASE_CM  150      // queue re-arm band: person ahead clearly left
#define SOUND_SPEED_CM_US 0.0343f

// === Alert pulse ===
#define PULSE_MS          1000     // 1 second trigger pulse

// === Consensus (distance noise filter) ===
#define HISTORY_SIZE      20
#define SIMILARITY_CM     10
#define MIN_AGREEMENT     2

// === State confirmation (distance side) ===
#define STATE_CONFIRM_N   3        // N consecutive readings to flip in/out state

// === Ultrasonic timing ===
#define PING_INTERVAL_MS  40
#define ECHO_TIMEOUT_US   25000
#define DEBUG_PRINT_MS    200

// ============================================================
//  >>>>>>>>>>  MODE-SWITCH TUNING  (change these!)  <<<<<<<<<<
// ============================================================
#define MODE_SWITCH_MS    2000     // <-- how long a condition must hold before mode commits (ms)
#define IMU_SAMPLE_MS     10       // fast IMU sampling for variance
#define IMU_WINDOW        32       // accel-magnitude samples in the variance window
#define VAR_HIGH          0.020f   // variance ABOVE this -> WALKING   (tolerance band top)
#define VAR_LOW           0.008f   // variance BELOW this -> QUEUING    (tolerance band bottom)
//   Between VAR_LOW and VAR_HIGH = dead-zone: hold current mode (no flip-flop)
//   Units are g^2 (accel magnitude is normalized to g). Calibrate with the
//   debug print: watch "var=" while walking vs standing still, then set
//   VAR_LOW just above your "still" number and VAR_HIGH just below "walking".
// ============================================================

// === Modes ===
enum Mode { MODE_WALKING, MODE_QUEUING };
volatile uint8_t currentMode = MODE_WALKING;   // start assuming movement (safer)

// === ISR shared state (ultrasonic) ===
volatile unsigned long echoStartUs = 0;
volatile unsigned long echoEndUs = 0;
volatile bool echoDone = false;

// === Distance buffer & timing ===
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

// === Inside/outside state (alert logic) ===
bool wasInside = false;            // currently flagged too-close
uint8_t insideCount = 0;
uint8_t outsideCount = 0;

// === IMU variance window ===
float accelWindow[IMU_WINDOW];
uint8_t accelIndex = 0;
bool accelFull = false;
unsigned long lastImuMs = 0;
float lastVar = 0.0f;

// === Mode-switch debounce ===
uint8_t candidateMode = MODE_WALKING;   // what the variance currently suggests
unsigned long candidateSinceMs = 0;     // when the candidate first appeared

void IRAM_ATTR echoISR() {
  if (digitalRead(ECHO_PIN) == HIGH) {
    echoStartUs = micros();        // rising edge: echo started
  } else {
    echoEndUs = micros();          // falling edge: echo ended
    echoDone = true;
  }
}

// ---------- MPU6050 raw access (no library needed) ----------
void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void mpuInit() {
  mpuWrite(0x6B, 0x00);   // PWR_MGMT_1: wake up
  mpuWrite(0x1C, 0x00);   // ACCEL_CONFIG: +-2g full scale
  delay(50);
}

// Reads accel, returns magnitude in g (1.0 = gravity at rest)
float mpuReadAccelMag() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                 // ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);
  if (Wire.available() < 6) return 1.0f;

  int16_t ax = (Wire.read() << 8) | Wire.read();
  int16_t ay = (Wire.read() << 8) | Wire.read();
  int16_t az = (Wire.read() << 8) | Wire.read();

  // +-2g range -> 16384 LSB/g
  float gx = ax / 16384.0f;
  float gy = ay / 16384.0f;
  float gz = az / 16384.0f;
  return sqrtf(gx * gx + gy * gy + gz * gz);
}

// Variance of the accel-magnitude window
float computeVariance() {
  uint8_t total = accelFull ? IMU_WINDOW : accelIndex;
  if (total < 2) return 0.0f;
  float mean = 0.0f;
  for (uint8_t i = 0; i < total; i++) mean += accelWindow[i];
  mean /= total;
  float var = 0.0f;
  for (uint8_t i = 0; i < total; i++) {
    float d = accelWindow[i] - mean;
    var += d * d;
  }
  return var / total;
}

// ---------- Distance consensus ----------
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

// ---------- IMU mode decision ----------
void updateMode() {
  unsigned long now = millis();
  if (now - lastImuMs < IMU_SAMPLE_MS) return;
  lastImuMs = now;

  // sample accel magnitude into the window
  accelWindow[accelIndex] = mpuReadAccelMag();
  accelIndex = (accelIndex + 1) % IMU_WINDOW;
  if (accelIndex == 0) accelFull = true;

  lastVar = computeVariance();

  // decide what the variance suggests right now (with dead-zone tolerance)
  uint8_t suggestion = currentMode;            // default: keep current
  if (lastVar >= VAR_HIGH)      suggestion = MODE_WALKING;
  else if (lastVar <= VAR_LOW)  suggestion = MODE_QUEUING;
  // else: in the tolerance band -> keep current mode

  // debounce: suggestion must persist MODE_SWITCH_MS before committing
  if (suggestion != currentMode) {
    if (suggestion != candidateMode) {
      candidateMode = suggestion;
      candidateSinceMs = now;
    } else if (now - candidateSinceMs >= MODE_SWITCH_MS) {
      currentMode = candidateMode;
      // reset alert state on any mode change so logic starts clean
      wasInside = false;
      insideCount = 0;
      outsideCount = 0;
      Serial.print("Mode switched to: ");
      Serial.println(currentMode == MODE_WALKING ? "WALKING" : "QUEUING");
    }
  } else {
    candidateMode = currentMode;   // suggestion matches current; cancel any pending switch
  }
}

// ---------- Alert decision ----------
void processReading(unsigned long dist) {
  history[historyIndex] = dist;
  historyIndex = (historyIndex + 1) % HISTORY_SIZE;
  if (historyIndex == 0) historyFull = true;

  lastDist = dist;
  lastAgreement = (dist != 0) ? countSimilar(dist) : 0;

  if (dist == 0) return;           // invalid reading: don't change state

  bool reliable = (lastAgreement >= MIN_AGREEMENT);
  bool readingInside = (dist < THRESHOLD_CM);

  // re-arm band depends on mode:
  //   WALKING -> small hysteresis (110cm): every new obstacle re-arms
  //   QUEUING -> wide release (150cm): only re-arm when person ahead clearly leaves
  unsigned long releaseCm = (currentMode == MODE_QUEUING)
                              ? QUEUE_RELEASE_CM
                              : (THRESHOLD_CM + HYSTERESIS_CM);
  bool readingOutside = (dist > releaseCm);

  if (readingInside && reliable) {
    insideCount++;
    outsideCount = 0;
    if (insideCount >= STATE_CONFIRM_N && !wasInside) {
      wasInside = true;
      fireTrigger();               // fire once on outside->inside crossing
    }
  } else if (readingOutside) {
    outsideCount++;
    insideCount = 0;
    if (outsideCount >= STATE_CONFIRM_N && wasInside) {
      wasInside = false;           // re-arm
    }
  }
  // between threshold and release band -> hold state (dead-zone)
}

void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(TRIGGER_PIN, LOW);

  for (uint8_t i = 0; i < HISTORY_SIZE; i++) history[i] = 0;
  for (uint8_t i = 0; i < IMU_WINDOW; i++) accelWindow[i] = 1.0f;

  Wire.begin(I2C_SDA, I2C_SCL);    // remap I2C: SDA=21, SCL=18
  mpuInit();

  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoISR, CHANGE);
  lastPingMs = millis();
  lastImuMs = millis();

  Serial.println("Ready - Mode: WALKING");
}

void loop() {
  unsigned long now = millis();

  // --- IMU: update mode ---
  updateMode();

  // --- Ultrasonic Task 1: new ping ---
  if (!waitingForEcho && (now - lastPingMs >= PING_INTERVAL_MS)) {
    lastPingMs = now;
    triggerPing();
  }

  // --- Task 2: finished echo ---
  if (waitingForEcho && echoDone) {
    waitingForEcho = false;
    unsigned long duration = echoEndUs - echoStartUs;
    unsigned long dist = (unsigned long)((duration * SOUND_SPEED_CM_US) / 2.0f);
    if (dist < 2 || dist > MAX_DISTANCE_CM) dist = 0;
    processReading(dist);
  }

  // --- Task 3: echo timeout ---
  if (waitingForEcho && (micros() - pingStartUs > ECHO_TIMEOUT_US)) {
    waitingForEcho = false;
    processReading(0);
  }

  // --- Task 4: drop alert pulse after PULSE_MS ---
  if (pinIsHigh && (long)(now - triggerHighUntilMs) >= 0) {
    digitalWrite(TRIGGER_PIN, LOW);
    pinIsHigh = false;
  }

  // --- Task 5: debug print (also your variance calibration readout) ---
  if (now - lastPrintMs >= DEBUG_PRINT_MS) {
    lastPrintMs = now;
    Serial.print(currentMode == MODE_WALKING ? "[WALKING] " : "[QUEUING] ");
    Serial.print("dist=");
    Serial.print(lastDist);
    Serial.print("cm inside=");
    Serial.print(wasInside ? "Y" : "N");
    Serial.print(" agree=");
    Serial.print(lastAgreement);
    Serial.print(" var=");
    Serial.println(lastVar, 4);
  }
}
