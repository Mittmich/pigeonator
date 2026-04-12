#include "test_server_fixtures.hpp"
#include <sstream>
#include <cstring>

namespace fs = std::filesystem;

// ─── ServerFixture ───────────────────────────────────────────────────────────

ServerFixture::ServerFixture() {
    // Create unique temp directory using PID + timestamp
    std::ostringstream dir_name;
    dir_name << "/tmp/bh_test_" << getpid() << "_"
             << std::chrono::steady_clock::now().time_since_epoch().count();
    base_dir_ = dir_name.str();
    db_path_ = base_dir_ / "db";
    uploads_dir_ = base_dir_ / "uploads";

    fs::create_directories(db_path_);
    fs::create_directories(uploads_dir_);

    db_url_ = "sqlite:///" + (db_path_ / "test.db").string();

    run_alembic_migrations();
    start_uvicorn();
    wait_for_server();
}

ServerFixture::~ServerFixture() {
    if (server_pid_ > 0) {
        kill(server_pid_, SIGTERM);
        int status = 0;
        waitpid(server_pid_, &status, 0);
    }
    // Clean up temp directory
    std::error_code ec;
    fs::remove_all(base_dir_, ec);
}

std::string ServerFixture::url() const {
    return "http://127.0.0.1:" + std::to_string(port_);
}

fs::path ServerFixture::uploads_dir() const {
    return uploads_dir_;
}

void ServerFixture::run_alembic_migrations() {
    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("fork() failed for alembic");
    }
    if (pid == 0) {
        // Child: set env, chdir, exec uv
        setenv("database_url", db_url_.c_str(), 1);
        setenv("upload_folder", uploads_dir_.c_str(), 1);
        if (chdir(BIRDHUB_SERVER_DIR) != 0) {
            _exit(1);
        }
        execlp("arch", "arch", "-arm64", "uv", "run", "--with-requirements", "requirements.txt",
               "alembic", "upgrade", "head", nullptr);
        _exit(1); // exec failed
    }
    // Parent: wait for alembic to finish
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("alembic upgrade head failed (exit code "
                                 + std::to_string(WEXITSTATUS(status)) + ")");
    }
}

void ServerFixture::start_uvicorn() {
    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("fork() failed for uvicorn");
    }
    if (pid == 0) {
        // Child: set env, chdir, exec uv run uvicorn
        setenv("database_url", db_url_.c_str(), 1);
        setenv("upload_folder", uploads_dir_.c_str(), 1);
        if (chdir(BIRDHUB_SERVER_DIR) != 0) {
            _exit(1);
        }
        // Redirect stdout/stderr to /dev/null to keep test output clean
        freopen("/dev/null", "w", stdout);
        freopen("/dev/null", "w", stderr);

        std::string port_str = std::to_string(port_);
        execlp("arch", "arch", "-arm64", "uv", "run", "--with-requirements", "requirements.txt",
               "uvicorn", "app.main:app",
               "--host", "127.0.0.1",
               "--port", port_str.c_str(),
               nullptr);
        _exit(1); // exec failed
    }
    server_pid_ = pid;
}

void ServerFixture::wait_for_server(int max_retries, int retry_delay_ms) {
    httplib::Client cli(url());
    cli.set_connection_timeout(1);
    cli.set_read_timeout(1);

    for (int i = 0; i < max_retries; ++i) {
        auto res = cli.Get("/");
        if (res && res->status == 200) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
    }
    // Server didn't come up — kill it and throw
    if (server_pid_ > 0) {
        kill(server_pid_, SIGTERM);
        waitpid(server_pid_, nullptr, 0);
        server_pid_ = -1;
    }
    throw std::runtime_error("birdhub-server did not start within "
                             + std::to_string(max_retries * retry_delay_ms) + "ms");
}

// ─── BirdhubApiClient ────────────────────────────────────────────────────────

BirdhubApiClient::BirdhubApiClient(const std::string& server_url)
    : server_url_(server_url) {}

int BirdhubApiClient::count_detections(const std::string& start_iso8601) const {
    httplib::Client cli(server_url_);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(10);

    std::string path = "/detections/";
    if (!start_iso8601.empty()) {
        path += "?start=" + start_iso8601;
    }

    auto res = cli.Get(path);
    if (!res || res->status != 200) {
        return -1;
    }

    auto j = nlohmann::json::parse(res->body, nullptr, false);
    if (j.is_discarded() || !j.is_array()) {
        return -1;
    }
    return static_cast<int>(j.size());
}

int BirdhubApiClient::count_recordings() const {
    httplib::Client cli(server_url_);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(10);

    auto res = cli.Get("/recordings/");
    if (!res || res->status != 200) {
        return -1;
    }

    auto j = nlohmann::json::parse(res->body, nullptr, false);
    if (j.is_discarded() || !j.is_array()) {
        return -1;
    }
    return static_cast<int>(j.size());
}
