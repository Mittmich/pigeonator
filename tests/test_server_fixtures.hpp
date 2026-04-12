#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef BIRDHUB_SERVER_DIR
#error "BIRDHUB_SERVER_DIR must be defined at compile time"
#endif

/**
 * RAII fixture that starts a birdhub-server (via uv) in a subprocess.
 *
 * Constructor:
 *   1. Creates temp dirs for DB and uploads
 *   2. Runs alembic migrations (blocking)
 *   3. Starts uvicorn in a child process
 *   4. Polls GET / until server responds
 *
 * Destructor:
 *   Sends SIGTERM to child and waits for exit.
 */
class ServerFixture {
public:
    ServerFixture();
    ~ServerFixture();

    // No copy/move
    ServerFixture(const ServerFixture&) = delete;
    ServerFixture& operator=(const ServerFixture&) = delete;

    std::string url() const;
    std::filesystem::path uploads_dir() const;

private:
    void run_alembic_migrations();
    void start_uvicorn();
    void wait_for_server(int max_retries = 40, int retry_delay_ms = 250);

    std::filesystem::path base_dir_;
    std::filesystem::path db_path_;
    std::filesystem::path uploads_dir_;
    std::string db_url_;
    int port_ = 18765;
    pid_t server_pid_ = -1;
};

/**
 * Lightweight HTTP client for verifying server state in tests.
 */
class BirdhubApiClient {
public:
    explicit BirdhubApiClient(const std::string& server_url);

    /// Returns total number of detections, optionally filtered by start time (ISO8601).
    int count_detections(const std::string& start_iso8601 = "") const;

    /// Returns total number of recordings.
    int count_recordings() const;

private:
    std::string server_url_;
};
