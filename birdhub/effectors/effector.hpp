#pragma once

#include "events.hpp"
#include <set>
#include <queue>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <chrono>
#include <thread>

class Effector : public Subscriber {
public:
    Effector(
        std::vector<std::string> target_classes,
        std::chrono::seconds cooldown
    );
    virtual ~Effector() = default;

    std::set<EventType> listening_to() override;
    void start() override;
    void stop() override;
    void set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue) override;
    void notify(std::shared_ptr<Event> event) override;

    virtual void trigger(const Detection& detection) = 0;

protected:
    bool is_activation_allowed() const;
    void emit_effector_action(const std::string& action,
                              std::optional<std::map<std::string, std::string>> meta_data = std::nullopt);

    std::vector<std::string> target_classes;
    std::chrono::seconds cooldown;
    std::optional<Timestamp> last_activation;
    std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue;

private:
    bool is_target_class(const Detection& detection) const;
};
