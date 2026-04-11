#include "effector.hpp"
#include <algorithm>
#include <iostream>

Effector::Effector(
    std::vector<std::string> target_classes,
    std::chrono::seconds cooldown)
    : target_classes(std::move(target_classes)),
      cooldown(cooldown) {}

std::set<EventType> Effector::listening_to() {
    return {EventType::DETECTION};
}

void Effector::start() {}
void Effector::stop() {}

void Effector::set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> eq) {
    event_queue = eq;
}

void Effector::notify(std::shared_ptr<Event> event) {
    if (event->type != EventType::DETECTION) return;
    auto det_event = std::static_pointer_cast<DetectionEvent>(event);

    if (!is_activation_allowed()) return;

    for (const auto& detection : det_event->get_detections()) {
        if (is_target_class(detection)) {
            last_activation = now();
            trigger(detection);
            return;
        }
    }
}

bool Effector::is_activation_allowed() const {
    if (!last_activation.has_value()) return true;
    auto elapsed = now() - last_activation.value();
    return elapsed >= cooldown;
}

void Effector::emit_effector_action(
    const std::string& action,
    std::optional<std::map<std::string, std::string>> meta_data)
{
    if (event_queue) {
        event_queue->push(std::make_shared<EffectorActionEvent>(now(), action, meta_data));
    }
}

bool Effector::is_target_class(const Detection& detection) const {
    if (target_classes.empty()) return true;
    auto labels_opt = detection.get_labels();
    if (!labels_opt.has_value()) return false;
    for (const auto& label : labels_opt.value()) {
        if (std::find(target_classes.begin(), target_classes.end(), label) != target_classes.end()) {
            return true;
        }
    }
    return false;
}
