"""Ferramentas expostas ao motor cognitivo (``@tool``)."""

from langchain_core.tools import BaseTool

from src.tools.home import control_device
from src.tools.search import web_search
from src.tools.weather import get_weather

ALL_TOOLS: list[BaseTool] = [get_weather, web_search, control_device]

__all__ = ["ALL_TOOLS", "get_weather", "web_search", "control_device"]
