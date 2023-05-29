"""Effectors that can be used to deter birds"""
from abc import ABC, abstractmethod

class Effector(ABC):

    @abstractmethod
    def activate(self) -> None:
        """Activate the effector"""
        raise NotImplementedError
    
    @abstractmethod
    def deactivate(self) -> None:
        """Deactivate the effector"""
        raise NotImplementedError
    
    @abstractmethod
    def is_active(self) -> bool:
        """Check if the effector is active"""
        raise NotImplementedError