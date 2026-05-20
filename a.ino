#define TRIG_PIN        18
#define ECHO_PIN        5
#define TRIGGER_PIN     23
#define BUTTON_PIN      19

#define MAX_DISTANCE_CM   400
#define THRESHOLD_CM      100
#define PULSE_MS          1000      // 改成 1s
#define SOUND_SPEED_CM_US 0.0343f

// === Consensus ===
#define HISTORY_SIZE      20
#define SIMILARITY_CM     10
#define MIN_AGREEMENT     2

// === State confirmation (NEW: 用几个读数确认进/出状态，抗噪) ===
#define STATE_CONFIRM_N   3         // 连续3次确认才改变 inside/outside 状态
#define HYSTERESIS_CM     10        // 迟滞，避免在阈值边缘抖动

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

// === Inside/outside state (NEW) ===
bool wasInside = false;             // 当前是否处于阈值内
uint8_t insideCount = 0;            // 连续判定为 inside 的次数
uint8_t outsideCount = 0;           // 连续判定为 outside 的次数

// === Button state ===
bool lastButtonState = HIGH;
bool buttonStableState = HIGH;
unsigned long lastDebounceMs = 0;

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

  // 无效读数：不改变状态，但累积 outside（可能人走了）
  if (dist == 0) {
    // 可选：把无读数当作"可能离开"，慢慢累积
    return;
  }

  // 用迟滞(hysteresis)判断 inside / outside
  // 进入需要 < THRESHOLD，离开需要 > THRESHOLD + 迟滞，避免边缘抖动
  bool readingInside  = (dist < THRESHOLD_CM);
  bool readingOutside = (dist > THRESHOLD_CM + HYSTERESIS_CM);

  // consensus 不够时不参与状态判定
  bool reliable = (lastAgreement >= MIN_AGREEMENT);

  if (readingInside && reliable) {
    insideCount++;
    outsideCount = 0;
    // 连续确认进入，且之前在外面 → 跨越边沿，触发
    if (insideCount >= STATE_CONFIRM_N && !wasInside) {
      wasInside = true;
      fireTrigger();          // 只在"从外进内"的瞬间触发
    }
  } else if (readingOutside) {
    outsideCount++;
    insideCount = 0;
    // 连续确认离开 → 重置状态，下次进入可再触发
    if (outsideCount >= STATE_CONFIRM_N && wasInside) {
      wasInside = false;
    }
  }
  // 在 THRESHOLD ~ THRESHOLD+HYSTERESIS 之间：保持当前状态不变（迟滞区）

  // STRICT 模式：覆盖上面逻辑，只要在阈值内且 consensus 够就每次触发
  if (currentMode == MODE_STRICT && readingInside && reliable) {
    fireTrigger();
  }
}

void checkButton() {
  bool reading = digitalRead(BUTTON_PIN);
  if (reading != lastButtonState) {
    lastDebounceMs = millis();
  }
  if ((millis() - lastDebounceMs) > DEBOUNCE_MS) {
    if (reading != buttonStableState) {
      buttonStableState = reading;
      if (buttonStableState == LOW) {
        currentMode = (currentMode + 1) % MODE_COUNT;
        wasInside = false;       // 切模式重置状态
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

  // Task 5: serial print
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
