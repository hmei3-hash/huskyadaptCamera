// ============================================================
// BASELINE VERSION — Threshold Only (for performance testing)
// ============================================================
// Stripped: consensus, history buffer, hysteresis, state
// confirmation, mode switching, button.
// Purpose: measure raw latency, power, CPU as a reference
// point to compare against the full algorithm.
// ============================================================

#define TRIG_PIN        18
#define ECHO_PIN        5
#define TRIGGER_PIN     23

#define MAX_DISTANCE_CM   400
#define THRESHOLD_CM      100
#define PULSE_MS          1000      // 1 second trigger pulse
#define SOUND_SPEED_CM_US 0.0343f

// === Timing ===
#define PING_INTERVAL_MS  40
#define ECHO_TIMEOUT_US   25000
#define DEBUG_PRINT_MS    200

// === ISR shared state ===
volatile unsigned long echoStartUs = 0;
volatile unsigned long echoEndUs = 0;
volatile bool echoDone = false;

// === Timing ===
unsigned long lastPingMs = 0;
unsigned long pingStartUs = 0;
bool waitingForEcho = false;

unsigned long triggerHighUntilMs = 0;
bool pinIsHigh = false;

unsigned long lastPrintMs = 0;
unsigned long lastDist = 0;

// === Inside/outside state (single variable) ===
bool wasInside = false;

void IRAM_ATTR echoISR() {
  if (digitalRead(ECHO_PIN) == HIGH) {
    echoStartUs = micros();         // Rising edge: echo pulse started
  } else {
    echoEndUs = micros();           // Falling edge: echo pulse ended
    echoDone = true;
  }
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
  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(TRIGGER_PIN, LOW);
  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoISR, CHANGE);
  lastPingMs = millis();
  Serial.println("Ready - BASELINE (threshold only)");
}

void processReading(unsigned long dist) {
  lastDist = dist;

  if (dist == 0) return;

  if (dist < THRESHOLD_CM) {
    if (!wasInside) {
      wasInside = true;
      fireTrigger();
    }
  } else {
    wasInside = false;
  }
}

void loop() {
  unsigned long now = millis();

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

  // Task 5: debug print
  if (now - lastPrintMs >= DEBUG_PRINT_MS) {
    lastPrintMs = now;
    Serial.print("[BASE] dist=");
    Serial.print(lastDist);
    Serial.print("cm inside=");
    Serial.println(wasInside ? "Y" : "N");
  }
}
