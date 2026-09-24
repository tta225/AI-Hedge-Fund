"""Read-only HTTP surface. Optional extra: ``pip install 'axiom[service]'``."""

from axiom.service.api import create_app, service_available

__all__ = ["create_app", "service_available"]
