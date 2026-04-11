#pragma once

#include "effector.hpp"

class MockEffector : public Effector {
public:
    MockEffector(
        std::vector<std::string> target_classes,
        std::chrono::seconds cooldown = std::chrono::seconds(10)
    );
    ~MockEffector() override = default;

    void trigger(const Detection& detection) override;
};
