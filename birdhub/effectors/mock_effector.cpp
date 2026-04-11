#include "mock_effector.hpp"
#include <iostream>

MockEffector::MockEffector(
    std::vector<std::string> target_classes,
    std::chrono::seconds cooldown)
    : Effector(std::move(target_classes), cooldown) {}

void MockEffector::trigger(const Detection& detection) {
    auto labels_opt = detection.get_labels();
    std::string label = (labels_opt.has_value() && !labels_opt->empty())
                        ? labels_opt->front()
                        : "unknown";
    std::cout << "[MockEffector] Activated for class: " << label << std::endl;

    std::map<std::string, std::string> meta{{"triggered_class", label}};
    emit_effector_action("effect_activated", meta);
}
