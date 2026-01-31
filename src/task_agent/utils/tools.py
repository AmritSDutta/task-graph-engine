from langchain_tavily import TavilySearch


def get_web_search_tool():
    return TavilySearch(
        max_results=5,
        topic="general",
    )
