/**
 * =============================================================
 *  BLE CLIENT — Reads from 2 Sensor Servers
 *  Board: ESP32
 *  Library: NimBLE-Arduino
 * 
 *  Scans for "TempSensor_1" and "HumSensor_2", connects to
 *  both, subscribes to their NOTIFY characteristics, and
 *  prints incoming sensor data to Serial.
 * =============================================================
 */

#include "NimBLEDevice.h"

// -----------------------------------------------------------
// UUIDs — must match the servers exactly
// -----------------------------------------------------------
#define SERVER1_SERVICE_UUID  "A1B2C3D4-E5F6-7890-ABCD-111111111111"
#define SERVER1_CHAR_UUID     "A1B2C3D4-E5F6-7890-ABCD-AABBCCDD0001"

#define SERVER2_SERVICE_UUID  "A1B2C3D4-E5F6-7890-ABCD-222222222222"
#define SERVER2_CHAR_UUID     "A1B2C3D4-E5F6-7890-ABCD-AABBCCDD0002"

// -----------------------------------------------------------
// State tracking for each server connection
// -----------------------------------------------------------
struct ServerInfo {
    const char*    name;          // for logging
    const char*    serviceUUID;
    const char*    charUUID;
    NimBLEClient*  pClient;
    bool           connected;
    bool           discovered;    // found during scan
    NimBLEAddress  address;       // stored after scan
};

ServerInfo servers[2] = {
    { "TempSensor_1", SERVER1_SERVICE_UUID, SERVER1_CHAR_UUID, nullptr, false, false, NimBLEAddress("") },
    { "HumSensor_2",  SERVER2_SERVICE_UUID, SERVER2_CHAR_UUID, nullptr, false, false, NimBLEAddress("") },
};

// -----------------------------------------------------------
// Notification Callback
// Called automatically when a server sends a notification.
// The `pRemoteCharacteristic` lets us identify which server
// sent it via its service UUID.
// -----------------------------------------------------------
void notifyCallback(NimBLERemoteCharacteristic* pChar, uint8_t* pData, size_t length, bool isNotify) {
    // Build a readable string from the raw bytes
    std::string value((char*)pData, length);

    // Identify source by characteristic UUID
    std::string charUUID = pChar->getUUID().toString();

    if (charUUID == NimBLEUUID(SERVER1_CHAR_UUID).toString()) {
        Serial.printf("[Client] *** Temperature (Server 1): %s °C\n", value.c_str());
    } else if (charUUID == NimBLEUUID(SERVER2_CHAR_UUID).toString()) {
        Serial.printf("[Client] *** Humidity    (Server 2): %s %%\n", value.c_str());
    } else {
        Serial.printf("[Client] Unknown notification from char: %s  value: %s\n",
                      charUUID.c_str(), value.c_str());
    }
}

// -----------------------------------------------------------
// Client Callbacks — handle disconnect per server slot
// -----------------------------------------------------------
class ClientCallbacks : public NimBLEClientCallbacks {
public:
    int serverIndex;  // which entry in servers[] this client belongs to

    ClientCallbacks(int idx) : serverIndex(idx) {}

    void onConnect(NimBLEClient* pClient) override {
        Serial.printf("[Client] Connected to %s\n", servers[serverIndex].name);
    }

    void onDisconnect(NimBLEClient* pClient, int reason) override {
        Serial.printf("[Client] Disconnected from %s (reason: %d) — will retry\n",
                      servers[serverIndex].name, reason);
        servers[serverIndex].connected = false;
    }
};

// -----------------------------------------------------------
// Scan & populate server addresses
// -----------------------------------------------------------
void scanForServers() {
    Serial.println("[Client] Scanning for servers...");

    NimBLEScan* pScan = NimBLEDevice::getScan();
    pScan->setActiveScan(true);

    // Scan until both servers are found or timeout (15s)
    NimBLEScanResults results = pScan->getResults(15 * 1000, false);

    for (int i = 0; i < results.getCount(); i++) {
        const NimBLEAdvertisedDevice* device = results.getDevice(i);

        for (int s = 0; s < 2; s++) {
            if (!servers[s].discovered &&
                device->isAdvertisingService(NimBLEUUID(servers[s].serviceUUID))) {

                servers[s].address   = device->getAddress();
                servers[s].discovered = true;
                Serial.printf("[Client] Found %s at %s\n",
                              servers[s].name,
                              device->getAddress().toString().c_str());
            }
        }
    }

    pScan->clearResults();
}

// -----------------------------------------------------------
// Connect to a single server and subscribe to notifications
// -----------------------------------------------------------
bool connectToServer(int idx) {
    ServerInfo& srv = servers[idx];

    if (!srv.discovered) {
        Serial.printf("[Client] %s not found in scan — skipping\n", srv.name);
        return false;
    }

    // Create a new client if needed
    if (srv.pClient == nullptr) {
        srv.pClient = NimBLEDevice::createClient();
        if (!srv.pClient) {
            Serial.printf("[Client] Failed to create client for %s\n", srv.name);
            return false;
        }
        srv.pClient->setCallbacks(new ClientCallbacks(idx));
    }

    Serial.printf("[Client] Connecting to %s...\n", srv.name);

    if (!srv.pClient->connect(srv.address)) {
        Serial.printf("[Client] Connection to %s FAILED\n", srv.name);
        return false;
    }

    // Get the service
    NimBLERemoteService* pService = srv.pClient->getService(srv.serviceUUID);
    if (pService == nullptr) {
        Serial.printf("[Client] Service not found on %s\n", srv.name);
        srv.pClient->disconnect();
        return false;
    }

    // Get the characteristic
    NimBLERemoteCharacteristic* pChar = pService->getCharacteristic(srv.charUUID);
    if (pChar == nullptr) {
        Serial.printf("[Client] Characteristic not found on %s\n", srv.name);
        srv.pClient->disconnect();
        return false;
    }

    // Subscribe to notifications
    if (pChar->canNotify()) {
        if (!pChar->subscribe(true, notifyCallback)) {
            Serial.printf("[Client] Failed to subscribe to %s notifications\n", srv.name);
            srv.pClient->disconnect();
            return false;
        }
        Serial.printf("[Client] Subscribed to notifications from %s\n", srv.name);
    } else {
        Serial.printf("[Client] %s characteristic does not support notify\n", srv.name);
        return false;
    }

    srv.connected = true;
    return true;
}

// -----------------------------------------------------------
// Setup
// -----------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("[Client] BLE Multi-Server Client starting...");

    NimBLEDevice::init("");

    // Scan and connect to both servers
    scanForServers();

    for (int i = 0; i < 2; i++) {
        connectToServer(i);
    }

    Serial.println("[Client] Setup complete. Waiting for notifications...");
}

// -----------------------------------------------------------
// Loop — reconnect any dropped servers
// -----------------------------------------------------------
void loop() {
    for (int i = 0; i < 2; i++) {
        if (!servers[i].connected) {
            Serial.printf("[Client] %s not connected — rescanning and retrying...\n", servers[i].name);

            // Reset discovery state so we re-scan for it
            servers[i].discovered = false;
            scanForServers();
            connectToServer(i);
        }
    }

    // Nothing else needed — notifications arrive via callback
    delay(5000);
}
