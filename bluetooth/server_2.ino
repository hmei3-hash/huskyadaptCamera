/**
 * =============================================================
 *  BLE SERVER 2 — Humidity Sensor
 *  Board: ESP32
 *  Library: NimBLE-Arduino
 * 
 *  This device advertises as "HumSensor_2" and notifies the
 *  connected client with a humidity reading every 2 seconds.
 * =============================================================
 */

#include "NimBLEDevice.h"

// -----------------------------------------------------------
// UUIDs — must match exactly on the client
// Server 2 has a DIFFERENT Service UUID than Server 1 so the
// client can distinguish which server is which.
// -----------------------------------------------------------
#define SERVICE_UUID        "A1B2C3D4-E5F6-7890-ABCD-222222222222"
#define CHAR_HUM_UUID       "A1B2C3D4-E5F6-7890-ABCD-AABBCCDD0002"

// -----------------------------------------------------------
// Globals
// -----------------------------------------------------------
NimBLEServer*         pServer         = nullptr;
NimBLECharacteristic* pHumChar        = nullptr;
bool                  clientConnected = false;

// Simulate a humidity reading (replace with real sensor code)
float readHumidity() {
    // TODO: Replace with actual sensor read, e.g. DHT22, SHT31, etc.
    static float base = 55.0f;
    base += ((float)(random(-10, 11))) / 10.0f;
    base = constrain(base, 30.0f, 90.0f);
    return base;
}

// -----------------------------------------------------------
// Server Callbacks — track connection state
// -----------------------------------------------------------
class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) override {
        Serial.println("[Server2] Client connected!");
        clientConnected = true;
    }

    void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) override {
        Serial.println("[Server2] Client disconnected — restarting advertising...");
        clientConnected = false;
        NimBLEDevice::startAdvertising();
    }
};

// -----------------------------------------------------------
// Setup
// -----------------------------------------------------------
void setup() {
    Serial.begin(115200);
    Serial.println("[Server2] Starting BLE Server 2...");

    NimBLEDevice::init("HumSensor_2");

    // Create server and attach callbacks
    pServer = NimBLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    // Create service
    NimBLEService* pService = pServer->createService(SERVICE_UUID);

    // Create humidity characteristic with NOTIFY property
    pHumChar = pService->createCharacteristic(
        CHAR_HUM_UUID,
        NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY
    );

    pHumChar->setValue("0.0");

    pService->start();

    // Start advertising
    NimBLEAdvertising* pAdvertising = NimBLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setName("HumSensor_2");
    pAdvertising->start();

    Serial.println("[Server2] Advertising started. Waiting for client...");
}

// -----------------------------------------------------------
// Loop — send humidity notification every 2 seconds
// -----------------------------------------------------------
void loop() {
    if (clientConnected) {
        float hum = readHumidity();

        char buf[16];
        snprintf(buf, sizeof(buf), "%.2f", hum);

        pHumChar->setValue(buf);
        pHumChar->notify();

        Serial.printf("[Server2] Notified humidity: %s %%\n", buf);
    }

    delay(2000);
}
