#define TRIG_PIN        18
#define ECHO_PIN        5
#define TRIGGER_PIN     23

#define MAX_DISTANCE_CM   400
#define THRESHOLD_CM      100
#define PULSE_MS          500
#define SOUND_SPEED_CM_US 0.0343f

// === Consensus parameters ===
#define HISTORY_SIZE      20
#define SIMILARITY_CM     10
#define MIN_AGREEMENT     2

// === Stillness detection (NEW) ===
#define STILLNESS_WINDOW    30      // 检查最近30次读数
#define STILLNESS_RANGE_CM  8       // 波动范围 ≤8cm 视为静止
#define STILLNESS_MIN_COUNT 25      // 30次里至少25次落在范围内

// === Timing ===
#define PING_INTERVAL_MS  40        // 提高到 ~25Hz
#define ECHO_TIMEOUT_US   25000     // 缩短到 25ms (~4m)
#define DEBUG_PRINT_MS    200       // Serial 打印降频，减少阻塞

// === ISR shared state ===
volatile unsigned long echoStartUs = 0;
volatile unsigned long echoEndUs = 0;
volatile bool echoDone = false;

// === Buffer & timing state ===
unsigned long history[HISTORY_SIZE];
uint8_t historyIndex = 0;
bool historyFull = false;

// === Stillness history (separate, longer window) ===
unsigned long stillHistory[STILLNESS_WINDOW];
uint8_t stillIndex = 0;
bool stillHistoryFull = false;

unsigned long lastPingMs = 0;
unsigned long pingStartUs = 0;
bool waitingForEcho = false;

unsigned long triggerHighUntilMs = 0;
bool pinIsHigh = false;

unsigned long lastPrintMs = 0;
unsigned long lastDist = 0;
uint8_t lastAgreement = 0;
bool lastStill = false;

void IRAM_ATTR echoISR() {
  if (digitalRead(ECHO_PIN) == HIGH) {
    echoStartUs = micros();
  } else {
    echoEndUs = micros();
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

// === NEW: 判断物体是否静止 ===
bool isStill(unsigned long currentDist) {
  if (currentDist == 0) return false;
  uint8_t total = stillHistoryFull ? STILLNESS_WINDOW : stillIndex;
  if (total < STILLNESS_MIN_COUNT) return false;  // 数据不够，先不判定
  
  uint8_t inRange = 0;
  for (uint8_t i = 0; i < total; i++) {
    if (stillHistory[i] == 0) continue;
    long diff = (long)stillHistory[i] - (long)currentDist;
    if (diff < 0) diff = -diff;
    if (diff <= STILLNESS_RANGE_CM) inRange++;
  }
  return inRange >= STILLNESS_MIN_COUNT;
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

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(TRIGGER_PIN, LOW);
  for (uint8_t i = 0; i < HISTORY_SIZE; i++) history[i] = 0;
  for (uint8_t i = 0; i < STILLNESS_WINDOW; i++) stillHistory[i] = 0;
  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoISR, CHANGE);
  lastPingMs = millis();
  Serial.println("Ready");
}

void processReading(unsigned long dist) {
  // 短窗口历史（用于 consensus）
  history[historyIndex] = dist;
  historyIndex = (historyIndex + 1) % HISTORY_SIZE;
  if (historyIndex == 0) historyFull = true;

  // 长窗口历史（用于 stillness）
  stillHistory[stillIndex] = dist;
  stillIndex = (stillIndex + 1) % STILLNESS_WINDOW;
  if (stillIndex == 0) stillHistoryFull = true;

  lastDist = dist;
  lastAgreement = 0;
  lastStill = false;

  if (dist != 0 && dist < THRESHOLD_CM) {
    uint8_t agreement = countSimilar(dist);
    lastAgreement = agreement;
    bool still = isStill(dist);
    lastStill = still;

    // 只有 consensus 通过 且 不是静止物体 才触发
    if (agreement >= MIN_AGREEMENT && !still) {
      digitalWrite(TRIGGER_PIN, HIGH);
      pinIsHigh = true;
      triggerHighUntilMs = millis() + PULSE_MS;
    }
  }
}

void loop() {
  unsigned long now = millis();

  // Task 1: ping
  if (!waitingForEcho && (now - lastPingMs >= PING_INTERVAL_MS)) {
    lastPingMs = now;
    triggerPing();
  }

  // Task 2: echo done
  if (waitingForEcho && echoDone) {
    waitingForEcho = false;
    unsigned long duration = echoEndUs - echoStartUs;
    unsigned long dist = (unsigned long)((duration * SOUND_SPEED_CM_US) / 2.0f);
    if (dist < 2 || dist > MAX_DISTANCE_CM) dist = 0;
    processReading(dist);
  }

  // Task 3: echo timeout
  if (waitingForEcho && (micros() - pingStartUs > ECHO_TIMEOUT_US)) {
    waitingForEcho = false;
    processReading(0);
  }

  // Task 4: drop trigger pin
  if (pinIsHigh && (long)(now - triggerHighUntilMs) >= 0) {
    digitalWrite(TRIGGER_PIN, LOW);
    pinIsHigh = false;
  }

  // Task 5: Serial print (低频，避免阻塞主循环)
  if (now - lastPrintMs >= DEBUG_PRINT_MS) {
    lastPrintMs = now;
    Serial.print("dist=");
    Serial.print(lastDist);
    Serial.print("cm agree=");
    Serial.print(lastAgreement);
    Serial.print("/");
    Serial.print(HISTORY_SIZE);
    if (lastStill) Serial.print(" [STILL-IGNORED]");
    else if (lastAgreement >= MIN_AGREEMENT && lastDist < THRESHOLD_CM) Serial.print(" [TRIGGERED]");
    Serial.println();
  }
}
