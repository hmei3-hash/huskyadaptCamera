#define TRIG_PIN        18
#define ECHO_PIN        5
#define TRIGGER_PIN     23

#define MAX_DISTANCE_CM   400
#define THRESHOLD_CM      100      // Social distance threshold (1m)
#define PULSE_MS          500      // How long to hold D23 HIGH
#define SOUND_SPEED_CM_US 0.0343f

// === Consensus parameters ===
#define HISTORY_SIZE      20
#define SIMILARITY_CM     10
#define MIN_AGREEMENT     2

// === Timing ===
#define PING_INTERVAL_MS  60       // 60ms between pings (~16 Hz)
#define ECHO_TIMEOUT_US   30000    // 30ms = ~5m worth of headroom

// === ISR shared state (must be volatile) ===
volatile unsigned long echoStartUs = 0;
volatile unsigned long echoEndUs = 0;
volatile bool echoDone = false;

// === Buffer & timing state ===
unsigned long history[HISTORY_SIZE];
uint8_t historyIndex = 0;
bool historyFull = false;

unsigned long lastPingMs = 0;
unsigned long pingStartUs = 0;
bool waitingForEcho = false;

unsigned long triggerHighUntilMs = 0;
bool pinIsHigh = false;

// === ISR: called on ECHO pin change ===
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
  // Send 10us pulse on TRIG
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  echoDone = false;
  pingStartUs = micros();
  waitingForEcho = true;
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(TRIGGER_PIN, LOW);

  for (uint8_t i = 0; i < HISTORY_SIZE; i++) history[i] = 0;

  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoISR, CHANGE);
  lastPingMs = millis();
  Serial.println("Ready");
}

void processReading(unsigned long dist) {
  history[historyIndex] = dist;
  historyIndex = (historyIndex + 1) % HISTORY_SIZE;
  if (historyIndex == 0) historyFull = true;

  Serial.print("dist=");
  Serial.print(dist);
  Serial.print("cm ");

  if (dist != 0 && dist < THRESHOLD_CM) {
    uint8_t agreement = countSimilar(dist);
    Serial.print("agree=");
    Serial.print(agreement);
    Serial.print("/");
    Serial.print(HISTORY_SIZE);

    if (agreement >= MIN_AGREEMENT) {
      Serial.print(" [TRIGGERED]");
      digitalWrite(TRIGGER_PIN, HIGH);
      pinIsHigh = true;
      triggerHighUntilMs = millis() + PULSE_MS;
    }
  }
  Serial.println();
}

void loop() {
  unsigned long now = millis();

  // === Task 1: kick off a new ping every PING_INTERVAL_MS ===
  if (!waitingForEcho && (now - lastPingMs >= PING_INTERVAL_MS)) {
    lastPingMs = now;
    triggerPing();
  }

  // === Task 2: handle finished echo ===
  if (waitingForEcho && echoDone) {
    waitingForEcho = false;
    unsigned long duration = echoEndUs - echoStartUs;
    unsigned long dist = (unsigned long)((duration * SOUND_SPEED_CM_US) / 2.0f);
    if (dist < 2 || dist > MAX_DISTANCE_CM) dist = 0;  // mark invalid
    processReading(dist);
  }

  // === Task 3: timeout check (no echo received) ===
  if (waitingForEcho && (micros() - pingStartUs > ECHO_TIMEOUT_US)) {
    waitingForEcho = false;
    processReading(0);  // 0 = no reading
  }

  // === Task 4: non-blocking pulse drop ===
  if (pinIsHigh && (long)(now - triggerHighUntilMs) >= 0) {
    digitalWrite(TRIGGER_PIN, LOW);
    pinIsHigh = false;
  }
}
