#include "sound_effector.hpp"
#include <cstdlib>
#include <iostream>
#include <thread>

SoundEffector::SoundEffector(
    std::vector<std::string> target_classes,
    std::string sound_file,
    std::chrono::seconds cooldown)
    : Effector(std::move(target_classes), cooldown),
      sound_file(std::move(sound_file)) {}

void SoundEffector::trigger(const Detection& detection) {
    auto labels_opt = detection.get_labels();
    std::string label = (labels_opt.has_value() && !labels_opt->empty())
                        ? labels_opt->front()
                        : "unknown";
    std::cout << "[SoundEffector] Playing sound for class: " << label << std::endl;

    std::string file = sound_file;
#ifdef __APPLE__
    std::string cmd = "afplay \"" + file + "\"";
#else
    std::string cmd = "aplay \"" + file + "\"";
#endif

    // Detach so the event loop is never blocked
    std::thread([cmd]() {
        std::system(cmd.c_str());
    }).detach();

    std::map<std::string, std::string> meta{{"triggered_class", label}, {"sound_file", file}};
    emit_effector_action("effect_activated", meta);
}
