import json
import os
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from PyQt6.QtCore import QThread
from langchain_core.tracers import BaseTracer
from langgraph.graph import Graph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import START, END
try:
    # The repo vendors `langchain_openai/`, but when running `python agentor/gui3.py`,
    # sys.path[0] becomes agentor/, which may hide repo-root packages.
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Annotated
from langchain_core.callbacks import BaseCallbackHandler
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from wsi_tools import segment_image, extract_features, build_model, analyze_results, generate_report
from openai import OpenAI
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)
from uuid import UUID
from langchain_core.tracers.schemas import Run



class State(TypedDict):
    messages: Annotated[list, add_messages]


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read from environment variables first; if missing, fall back to a local secrets file.
    Default secrets file location: agentor/secrets.local.json
    You can also set PYPATHOMICS_SECRETS_PATH to override.
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        # fallback: local secrets file
        try:
            secrets = _load_local_secrets()
            v2 = secrets.get(name)
            if isinstance(v2, str) and v2.strip() != "":
                return v2
        except Exception:
            # ignore parsing errors and fall back to default
            pass
        return default
    return value


def _load_local_secrets() -> dict:
    secrets_path = os.getenv("PYPATHOMICS_SECRETS_PATH", "").strip()
    if secrets_path:
        p = Path(secrets_path)
    else:
        p = Path(__file__).with_name("secrets.local.json")

    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # allow BOM / non-utf8 files by falling back
        data = json.loads(p.read_text(errors="ignore"))
    return data if isinstance(data, dict) else {}


def _build_llm(api_key=None, base_url=None, model_name=None) -> ChatOpenAI:
    """
    Build an LLM client (ChatOpenAI-compatible).

    Parameters:
    - api_key: API key (if None, read from environment/secrets)
    - base_url: Base URL (if None, read from environment/secrets or use default)
    - model_name: Model name (if None, read from environment/secrets or use default)
    
    Environment variables:
    - DASHSCOPE_API_KEY or OPENAI_API_KEY
    - OPENAI_BASE_URL (optional; defaults to DashScope OpenAI-compatible endpoint)
    - OPENAI_MODEL (optional; default: qwen3-max)
    """
    if not api_key:
        api_key = _get_env("DASHSCOPE_API_KEY") or _get_env("OPENAI_API_KEY")
    if not base_url:
        base_url = _get_env("OPENAI_BASE_URL",
                            "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not model_name:
        model_name = _get_env("OPENAI_MODEL", "qwen3-max")
    
    if api_key is None:
        raise RuntimeError(
            "No LLM API key detected. Please set environment variable DASHSCOPE_API_KEY or OPENAI_API_KEY, "
            "or configure it in GUI Setting Params > LLM Configuration."
        )
    return ChatOpenAI(openai_api_key=api_key,
                      model_name=model_name,
                      base_url=base_url)


def _build_openai_client(api_key=None, base_url=None) -> OpenAI:
    """
    Build an OpenAI client for report generation.
    
    Parameters:
    - api_key: API key (if None, read from environment/secrets)
    - base_url: Base URL (if None, read from environment/secrets or use default)
    """
    if not api_key:
        api_key = _get_env("DASHSCOPE_API_KEY") or _get_env("OPENAI_API_KEY")
    if not base_url:
        base_url = _get_env("OPENAI_BASE_URL",
                            "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if api_key is None:
        raise RuntimeError(
            "No OpenAI-compatible API key detected. Please set environment variable DASHSCOPE_API_KEY or OPENAI_API_KEY, "
            "or configure it in GUI Setting Params > LLM Configuration."
        )
    return OpenAI(api_key=api_key, base_url=base_url)

system_prompt = SystemMessage(content="""
You are a professional pathology analysis agent. You assist clinicians with pathology (WSI) workflows.

Important rules:
- If the user message contains a section that starts with `CONTEXT_JSON:` followed by JSON,
  you MUST use it to fill in tool arguments (paths and parameters). Do NOT ask the user
  to provide paths that already exist in CONTEXT_JSON. If something is missing there,
  then you may ask follow-up questions.
- CRITICAL: Always use the parameters from CONTEXT_JSON.setting_params when calling tools:
  * For extract_features: use setting_params.extract_features (n_workers, mag, wsi_num, etc.)
  * For build_model: use setting_params.build_model (n_workers, k_fold, repeats_num, feature selection, classifiers, etc.)
  * For analyze_results: use setting_params.analyze_results (top, top_signif, sources, clinical_col, category)
  DO NOT use default values when CONTEXT_JSON provides these parameters.
- If `CONTEXT_JSON.overrides.skip_segmentation` is true, you MUST NOT call the segmentation tool.
  Assume segmentation outputs already exist at `CONTEXT_JSON.ui_paths.seg_results_folder` and proceed.
- If `CONTEXT_JSON.overrides.skip_feature_extraction` is true, you MUST NOT call the feature extraction tool.
  Assume a feature matrix already exists at `CONTEXT_JSON.ui_paths.feature_matrix_csv` and proceed.
- If `CONTEXT_JSON.overrides.limit_to_modeling` is true, you MUST stop after model construction + cross-validation.
  Do NOT call the results analysis tool (analyze_results) and do NOT call the report generation tool.
- When the user asks to segment images, ONLY call the image segmentation tool. Default: cell segmentation.
- When the user asks to extract features, ONLY call the feature extraction tool (segmentation is required first).
- When the user asks to build a model, ONLY call the model construction tool (feature extraction is required first).
- When the user asks to analyze results / feature importance, ONLY call the results analysis tool ONCE.
  The 'sources' parameter can accept multiple databases separated by commas (e.g., "GO:All,KEGG,REAC").
  Do NOT call this tool multiple times for different sources.
- When the user asks to generate a report, ONLY call the report generation tool.
  If CONTEXT_JSON.outputs.last_cross_validation_dir is available, pass it as cross_validation_path parameter.
- When the user asks for a full pipeline, the order is:
  1) Segmentation -> 2) Feature extraction -> 3) Model construction -> 4) Result analysis -> 5) Report generation
- IMPORTANT: If the user's request is ONLY about transcriptomic/genomic analysis (keywords: "map phenotypes to transcriptomic/expression profiles",
  "reveal molecular drivers", "pathway activations", "differential expression", "gene enrichment", "DEG", "GSEA"),
  and does NOT explicitly mention segmentation/extraction/modeling, then DIRECTLY call the results analysis tool (analyze_results).
  Use paths from CONTEXT_JSON for cross_validation_dir (if available) or feature_matrix_csv. Do NOT re-run steps 1-3.
- When the user asks to query history, extract relevant information from prior messages and reply.
- IMPORTANT: In the same request, if a tool has already returned successfully, do NOT call the same tool again.
  Summarize based on the latest tool output and stop.
""")

tools = [
    segment_image,
    extract_features,
    build_model,
    analyze_results,
    generate_report
]


def route_tools(
        state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


class ToolCallbackHandler(BaseCallbackHandler):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals

    # def on_tool_start(self, serialized, input_str, **kwargs):
    #     tool_name = serialized.get('name', 'unknown_tool')
    #     # Use QueuedConnection to ensure thread safety.
    #     self.signals.tool_start.emit(tool_name, str(input_str))

    def on_tool_end(self, output, **kwargs):
        print(f"on_tool_end:{str(output)}")
        self.signals.tool_end.emit(str(output.content))


class WSIAgent:

    def __init__(self, gui=None, signals=None):
        """"""
        self.gui = gui
        # self.signals = signals
        self.checkpointer = InMemorySaver()
        # LangGraph's default recursion limit can be low (often ~25). When a single request
        # triggers multiple tool calls (or rare repeated calls), it may hit GRAPH_RECURSION_LIMIT.
        # Increase the limit for better robustness.
        self.config = {
            "configurable": {"thread_id": "1"},
            "callbacks": [self.gui.callback_handler],
            "return_intermediate_steps": True,
            "recursion_limit": 80,
        }
        # self.config = {"configurable": {"thread_id": "1"}}
        self.app = None
        self._init_error = None

    def _ensure_app(self):
        """Lazy initialization: let the GUI start first; create the LLM/Graph on first use."""
        if self.app is not None:
            return
        try:
            # Get LLM configuration from GUI if available
            api_key = None
            base_url = None
            model_name = None
            if self.gui and hasattr(self.gui, 'setting_views'):
                try:
                    llm_config = self.gui.setting_views.get_val().get("llm_config", {})
                    api_key = llm_config.get("api_key")
                    base_url = llm_config.get("base_url")
                    model_name = llm_config.get("agent_model")
                except Exception:
                    pass
            
            llm = _build_llm(api_key=api_key, base_url=base_url, model_name=model_name)
            agent = create_react_agent(
                llm, tools, prompt=system_prompt, checkpointer=self.checkpointer
            )
            workflow = Graph()
            workflow.add_node("agent", agent)
            tool_node = ToolNode(tools=tools)
            workflow.add_node("tools", tool_node)
            workflow.add_conditional_edges(
                "agent",
                route_tools,
                {"tools": "tools", END: END},
            )
            workflow.add_edge("tools", "agent")
            workflow.add_edge(START, "agent")
            self.app = workflow.compile(checkpointer=self.checkpointer)
        except Exception as e:
            self._init_error = str(e)
            raise

    def run_agent(self, query: str):
        self._ensure_app()

        memory_info = list(self.app.get_state_history(self.config))
        if memory_info:
            values = memory_info[0].values if memory_info[0].values else memory_info[1].values
            msg_list = values['agent']['messages']
            msg_list.append(HumanMessage(content=query))
        else:
            msg_list = [HumanMessage(content=query)]
        final_result = self.app.invoke({"messages": msg_list}, self.config)

        # print(final_result)
        # for msg in final_result["messages"]:
        #     if msg.type == "tool":
        #         try:
        #             self.gui.add_message(msg.content, is_ai=True)
        #         except Exception as e:
        #             self.gui.add_message(str(e), is_ai=True)
        # self.signals.tool_end.emit(final_result["messages"][-1].content)
        return final_result["messages"][-1].content

    def delete_memory(self):
        self.config['configurable']['thread_id'] = str(int(self.config['configurable']['thread_id']) + 1)


def parse_res(csv_path, prompt_str='Please analyze this CSV (FEA results) and provide a complete multi-dimensional interpretation. For important image features, explain which biological pathways/processes they are related to. Finally, generate a comprehensive analysis report.', api_key=None, base_url=None, report_model="qwen-long"):
    try:
        client = _build_openai_client(api_key=api_key, base_url=base_url)
        file_object = client.files.create(file=Path(csv_path), purpose="file-extract")
        completion = client.chat.completions.create(
            model=report_model,  # Use configurable model (default: qwen-long)
            messages=[
                {'role': 'system', 'content': f'fileid://{file_object.id}'},
                {'role': 'user', 'content': prompt_str}
            ]
        )
        return json.loads(completion.model_dump_json())['choices'][0]['message']['content']
    except Exception as e:
        return str(e)


def parse_res_v2(csv_path, txt_path=None, prompt_str='Based on the two files above, interpret the biological processes corresponding to the top four pathways one by one. For each pathway, describe whether upregulation/downregulation corresponds to high-risk or low-risk patients and the trends in their clinical characteristics. Finally, generate a complete English analysis report and keep the font size consistent between headings and body text.', api_key=None, base_url=None, report_model="qwen-long"):
    try:
        client = _build_openai_client(api_key=api_key, base_url=base_url)
        file_object = client.files.create(file=Path(csv_path), purpose="file-extract")
        msg_list = [
                {'role': 'system', 'content': f'fileid://{file_object.id}'}
            ]
        if txt_path:
            file_object1 = client.files.create(file=Path(txt_path), purpose="file-extract")
            msg_list.append({'role': 'system', 'content': f'fileid://{file_object1.id}'})
        msg_list.append({'role': 'user', 'content': prompt_str})
        completion = client.chat.completions.create(
            model=report_model,  # Use configurable model (default: qwen-long)
            messages=msg_list
        )
        return json.loads(completion.model_dump_json())['choices'][0]['message']['content']
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    client = _build_openai_client()
    file_object = client.files.create(file=Path('gseapy.gene_set.prerank.report.csv'), purpose="file-extract")
    file_object1 = client.files.create(file=Path('T_stage_statistics.txt'), purpose="file-extract")
    completion = client.chat.completions.create(
        model="qwen-long",  # Models: https://help.aliyun.com/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': f'fileid://{file_object.id}'},
            {'role': 'system', 'content': f'fileid://{file_object1.id}'},
            {'role': 'user',
             'content': "Based on the above two documents, parse the biological processes corresponding to the first four pathways one by one, as well as the distribution trends of high/low-risk patients associated with the upregulation/downregulation of each pathway and their clinical characteristics. For example, the upregulation/downregulation of the xxx pathway corresponds to high/low-risk patients, showing a trend of xxx clinical characteristics; finally, generate a comprehensive English analysis report with a consistent font size throughout the document."}
        ]
    )
    res = json.loads(completion.model_dump_json())['choices'][0]['message']['content']
    print(res)
