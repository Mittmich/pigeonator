#pragma once

#include "effector.hpp"
#include <string>

class SoundEffector : public Effector {
public:
    SoundEffector(
        std::vector<std::string> target_classes,
        std::string sound_file,
        std::chrono::seconds cooldown = std::chrono::seconds(10)
    );
    ~SoundEffector() override = default;

    void trigger(const Detection& detection) override;

private:
    std::string sound_file;
};
