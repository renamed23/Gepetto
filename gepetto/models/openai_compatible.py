import json
import threading
import functools
import urllib.request
import urllib.error
import ssl
import ida_kernwin
import gepetto.config
import gepetto.models.model_manager
from gepetto.models.base import LanguageModel

_ = gepetto.config._

DEFAULT_MODELS = ["default"]


class OpenAICompatible(LanguageModel):

    @staticmethod
    def get_menu_name() -> str:
        name = gepetto.config.get_config("OpenAICompatible", "NAME")
        return name if name else "OpenAICompatible"

    @staticmethod
    def supported_models():
        config_models = gepetto.config.get_config("OpenAICompatible", "MODELS")
        if config_models:
            try:
                return json.loads(config_models)
            except json.JSONDecodeError:
                return [m.strip() for m in config_models.split(",")]
        return DEFAULT_MODELS

    @staticmethod
    def is_configured_properly() -> bool:
        return bool(gepetto.config.get_config("OpenAICompatible", "API_KEY"))

    def __init__(self, model):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

        api_key = gepetto.config.get_config(
            "OpenAICompatible", "API_KEY", "OPENAI_COMPATIBLE_API_KEY"
        )
        if not api_key:
            raise ValueError(
                _("Please edit the configuration file to insert your {api_provider} API key!")
                .format(api_provider=OpenAICompatible.get_menu_name())
            )

        self.api_key = api_key

        base_url = gepetto.config.get_config(
            "OpenAICompatible", "BASE_URL", "OPENAI_COMPATIBLE_BASE_URL"
        )
        if not base_url:
            base_url = "https://api.openai.com/v1"

        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/chat/completions"

        handlers = []
        proxy = gepetto.config.get_config("Gepetto", "PROXY")
        if proxy:
            print(f"> [Gepetto] Using proxy: {proxy}")
            handlers.append(urllib.request.ProxyHandler(
                {"http": proxy, "https": proxy}))
        self.opener = urllib.request.build_opener(*handlers)
        self.ssl_context = ssl.create_default_context()

    def __str__(self):
        return self.model

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _execute_ui(self, fn):
        ida_kernwin.execute_sync(fn, ida_kernwin.MFF_WRITE)

    def _notify_error(self, cb, message, stream=False):
        print(f"\n> [Gepetto] ERROR: {message}")
        if not cb:
            return

        def _invoke():
            try:
                if stream:
                    cb({"error": message}, "error")
                else:
                    cb({"error": message})
            except TypeError:
                cb({"error": message})

        self._execute_ui(_invoke)

    def query_model(self, query, cb, stream=False, additional_model_options=None):
        if additional_model_options is None:
            additional_model_options = {}

        conversation = (
            [{"role": "user", "content": query}]
            if isinstance(query, str)
            else query
        )

        payload = {
            "model": self.model,
            "messages": conversation,
            "stream": stream,
            **additional_model_options
        }

        print(f"> [Gepetto] Requesting model: {self.model} (Stream: {stream})")
        data_bytes = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            self.url,
            data=data_bytes,
            headers=self._headers(),
            method="POST"
        )

        try:
            with self.opener.open(request, timeout=120) as response:
                print(
                    f"> [Gepetto] Connection established. Processing response...")

                if not stream:
                    raw = response.read().decode("utf-8")
                    data = json.loads(raw)

                    if "error" in data:
                        self._notify_error(cb, data["error"].get(
                            "message", str(data["error"])))
                        return

                    message = data["choices"][0]["message"]
                    print(
                        f"> [Gepetto] Full Response: {message.get('content')[:100]}...")

                    class Message:
                        def __init__(self, d):
                            self.role = d.get("role")
                            self.content = d.get("content")
                            self.tool_calls = d.get("tool_calls")

                    msg = Message(message)
                    self._execute_ui(lambda: cb(response=msg))

                    if "usage" in data:
                        self.input_tokens += data["usage"].get(
                            "prompt_tokens", 0)
                        self.output_tokens += data["usage"].get(
                            "completion_tokens", 0)

                else:
                    print(f"> [Gepetto] Streaming output start:\n" + "-"*30)
                    while True:
                        raw_line = response.readline()
                        if not raw_line:
                            break

                        line = raw_line.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            print("\n" + "-"*30 +
                                  "\n> [Gepetto] Streaming finished.")
                            self._execute_ui(lambda: cb({}, "stop"))
                            break

                        try:
                            chunk = json.loads(data_str)
                            if "error" in chunk:
                                self._notify_error(
                                    cb, str(chunk["error"]), stream=True)
                                return

                            delta = chunk.get("choices", [{}])[
                                0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                # 在 IDA 输出窗口实时打印内容（不换行）
                                print(content, end="")

                            finish_reason = chunk.get("choices", [{}])[
                                0].get("finish_reason")
                            self._execute_ui(functools.partial(
                                cb, delta, finish_reason))

                            if "usage" in chunk and chunk["usage"]:
                                self.input_tokens += chunk["usage"].get(
                                    "prompt_tokens", 0)
                                self.output_tokens += chunk["usage"].get(
                                    "completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

        except urllib.error.HTTPError as e:
            try:
                err_detail = e.read().decode("utf-8")
                msg = f"HTTP {e.code}: {err_detail}"
            except:
                msg = f"HTTP {e.code}: {e.reason}"
            self._notify_error(cb, msg, stream)
        except Exception as e:
            self._notify_error(cb, _(
                "General exception encountered while running the query: {error}").format(error=str(e)), stream)

    def query_model_async(self, query, cb, stream=False, additional_model_options=None):
        t = threading.Thread(
            target=self.query_model,
            args=(query, cb, stream, additional_model_options),
            daemon=True
        )
        t.start()


gepetto.models.model_manager.register_model(OpenAICompatible)
