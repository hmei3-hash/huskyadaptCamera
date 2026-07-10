// ============================================================
// FULL VERSION — IMU Fusion (Latency Test)
// ============================================================
// Pins: TRIG=5, ECHO=4, LED/BUZZER=6, SDA=40, SCL=41
// Algorithms: consensus + hysteresis + state confirmation
//             + IMU auto mode switching (WALKING/QUEUING)
// ============================================================

#include <Wire.h>

#define TRIG_PIN        5
#define ECHO_PIN        4
#define TRIGGER_PIN     6
#define I2C_SDA         40
#define I2C_SCL         41
#define MPU_ADDR        0x68

#define MAX_DISTANCE_CM   400
#define THRESHOLD_CM      100
#define HYSTERESIS_CM     10
#define QUEUE_RELEASE_CM  150
#define SOUND_SPEED_CM_US 0.0343f
#define PULSE_MS          1000

// === Consensus ===
#define HISTORY_SIZE      20
#define SIMILARITY_CM     10
#define MIN_AGREEMENT     2

// === State confirmation ===
#define STATE_CONFIRM_N   3

// === Timing ===
#define PING_INTERVAL_MS  40
#define ECHO_TIMEOUT_US   25000
#define DEBUG_PRINT_MS    200

// === IMU mode switching ===
#define MODE_SWITCH_MS    2000
#define IMU_SAMPLE_MS     10
#define IMU_WINDOW        32
#define VAR_HIGH          0.020f
#define VAR_LOW           0.008f

enum Mode { MODE_WALKING, MODE_QUEUING };
volatile uint8_t currentMode = MODE_WALKING;

// === ISR ===
volatile unsigned long echoStartUs = 0;
volatile unsigned long echoEndUs = 0;
volatile bool echoDone = false;

// === Distance buffer ===
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
bool wasInside = false;
uint8_t insideCount = 0;
uint8_t outsideCount = 0;

// === IMU ===
float accelWindow[IMU_WINDOW];
uint8_t accelIndex = 0;
bool accelFull = false;
unsigned long lastImuMs = 0;
float lastVar = 0.0f;

uint8_t candidateMode = MODE_WALKING;
unsigned long candidateSinceMs = 0;

void IRAM_ATTR echoISR() {
  if (digitalRead(ECHO_PIN) == HIGH) {
    echoStartUs = micros();
  } else {
    echoEndUs = micros();
    echoDone = true;
  }
}

// ---------- MPU6050 ----------
void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void mpuInit() {
  mpuWrite(0x6B, 0x00);
  mpuWrite(0x1C, 0x00);
  delay(50);
}

float mpuReadAccelMag() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);
  if (Wire.available() < 6) return 1.0f;
  int16_t ax = (Wire.read() << 8) | Wire.read();
  int16_t ay = (Wire.read() << 8) | Wire.read();
  int16_t az = (Wire.read() << 8) | Wire.read();
  float gx = ax / 16384.0f;
  float gy = ay / 16384.0f;
  float gz = az / 16384.0f;
  return sqrtf(gx * gx + gy * gy + gz * gz);
}

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

// ---------- Consensus ----------
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

  accelWindow[accelIndex] = mpuReadAccelMag();
  accelIndex = (accelIndex + 1) % IMU_WINDOW;
  if (accelIndex == 0) accelFull = true;

  lastVar = computeVariance();

  uint8_t suggestion = currentMode;
  if (lastVar >= VAR_HIGH)      suggestion = MODE_WALKING;
  else if (lastVar <= VAR_LOW)  suggestion = MODE_QUEUING;

  if (suggestion != currentMode) {
    if (suggestion != candidateMode) {
      candidateMode = suggestion;
      candidateSinceMs = now;
    } else if (now - candidateSinceMs >= MODE_SWITCH_MS) {
      currentMode = candidateMode;
      wasInside = false;
      insideCount = 0;
      outsideCount = 0;
      Serial.print("Mode switched to: ");
      Serial.println(currentMode == MODE_WALKING ? "WALKING" : "QUEUING");
    }
  } else {
    candidateMode = currentMode;
  }
}

// ---------- Alert decision ----------
void processReading(unsigned long dist, unsigned long echoReceivedUs) {
  history[historyIndex] = dist;
  historyIndex = (historyIndex + 1) % HISTORY_SIZE;
  if (historyIndex == 0) historyFull = true;

  lastDist = dist;
  lastAgreement = (dist != 0) ? countSimilar(dist) : 0;

  if (dist == 0) return;

  bool reliable = (lastAgreement >= MIN_AGREEMENT);
  bool readingInside = (dist < THRESHOLD_CM);

  unsigned long releaseCm = (currentMode == MODE_QUEUING)
                              ? QUEUE_RELEASE_CM
                              : (THRESHOLD_CM + HYSTERESIS_CM);
  bool readingOutside = (dist > releaseCm);

  if (readingInside && reliable) {
    insideCount++;
    outsideCount = 0;
    if (insideCount >= STATE_CONFIRM_N && !wasInside) {
      wasInside = true;

      unsigned long beforeUs = micros();
      fireTrigger();
      unsigned long afterUs = micros();

      unsigned long procUs = afterUs - echoReceivedUs;
      unsigned long e2eUs  = afterUs - pingStartUs;

      Serial.print("[LATENCY] dist=");
      Serial.print(dist);
      Serial.print("cm mode=");
      Serial.print(currentMode == MODE_WALKING ? "W" : "Q");
      Serial.print(" agree=");
      Serial.print(lastAgreement);
      Serial.print(" proc=");
      Serial.print(procUs);
      Serial.print("us e2e=");
      Serial.print(e2eUs);
      Serial.println("us");
    }
  } else if (readingOutside) {
    outsideCount++;
    insideCount = 0;
    if (outsideCount >= STATE_CONFIRM_N && wasInside) {
      wasInside = false;
    }
  }
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

  Wire.begin(I2C_SDA, I2C_SCL);
  mpuInit();

  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoISR, CHANGE);
  lastPingMs = millis();
  lastImuMs = millis();

  Serial.println("Ready - FULL IMU (latency test)");
  Serial.println("FORMAT: [LATENCY] dist=XXcm mode=W/Q agree=XX proc=XXus e2e=XXus");
}

void loop() {
  unsigned long now = millis();

  updateMode();

  if (!waitingForEcho && (now - lastPingMs >= PING_INTERVAL_MS)) {
    lastPingMs = now;
    triggerPing();
  }

  if (waitingForEcho && echoDone) {
    waitingForEcho = false;
    unsigned long echoReceivedUs = micros();
    unsigned long duration = echoEndUs - echoStartUs;
    unsigned long dist = (unsigned long)((duration * SOUND_SPEED_CM_US) / 2.0f);
    if (dist < 2 || dist > MAX_DISTANCE_CM) dist = 0;
    processReading(dist, echoReceivedUs);
  }

  if (waitingForEcho && (micros() - pingStartUs > ECHO_TIMEOUT_US)) {
    waitingForEcho = false;
    processReading(0, 0);
  }

  if (pinIsHigh && (long)(now - triggerHighUntilMs) >= 0) {
    digitalWrite(TRIGGER_PIN, LOW);
    pinIsHigh = false;
  }

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
