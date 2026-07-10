// ============================================================
// BASELINE — Threshold Only (Latency Test)
// ============================================================
// Pins: TRIG=5, ECHO=4, LED/BUZZER=6
// No algorithms. Pure threshold detection.
// ============================================================

#define TRIG_PIN        5
#define ECHO_PIN        4
#define TRIGGER_PIN     6

#define MAX_DISTANCE_CM   400
#define THRESHOLD_CM      100
#define PULSE_MS          1000
#define SOUND_SPEED_CM_US 0.0343f

#define PING_INTERVAL_MS  40
#define ECHO_TIMEOUT_US   25000
#define DEBUG_PRINT_MS    200

volatile unsigned long echoStartUs = 0;
volatile unsigned long echoEndUs = 0;
volatile bool echoDone = false;

unsigned long lastPingMs = 0;
unsigned long pingStartUs = 0;
bool waitingForEcho = false;

unsigned long triggerHighUntilMs = 0;
bool pinIsHigh = false;

unsigned long lastPrintMs = 0;
unsigned long lastDist = 0;

bool wasInside = false;

void IRAM_ATTR echoISR() {
  if (digitalRead(ECHO_PIN) == HIGH) {
    echoStartUs = micros();
  } else {
    echoEndUs = micros();
    echoDone = true;
  }
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

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(TRIGGER_PIN, LOW);
  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoISR, CHANGE);
  lastPingMs = millis();
  Serial.println("Ready - BASELINE (threshold only)");
  Serial.println("FORMAT: [LATENCY] dist=XXcm proc=XXus e2e=XXus");
}

void processReading(unsigned long dist, unsigned long echoReceivedUs) {
  lastDist = dist;

  if (dist == 0) return;

  if (dist < THRESHOLD_CM) {
    if (!wasInside) {
      wasInside = true;

      unsigned long beforeUs = micros();
      fireTrigger();
      unsigned long afterUs = micros();

      unsigned long procUs = afterUs - echoReceivedUs;
      unsigned long e2eUs  = afterUs - pingStartUs;

      Serial.print("[LATENCY] dist=");
      Serial.print(dist);
      Serial.print("cm proc=");
      Serial.print(procUs);
      Serial.print("us e2e=");
      Serial.print(e2eUs);
      Serial.println("us");
    }
  } else {
    wasInside = false;
  }
}

void loop() {
  unsigned long now = millis();

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
    Serial.print("[BASE] dist=");
    Serial.print(lastDist);
    Serial.print("cm inside=");
    Serial.println(wasInside ? "Y" : "N");
  }
}
