#!/usr/bin/env bash

# This file is sourced by worker launchers after .env. Keep it side-effect free
# apart from exporting the selected LLM client configuration.

_em2mem_profile_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_em2mem_profile_file="${EM2MEM_LLM_PROFILE_FILE:-${_em2mem_profile_root}/runtime/config/llm_profile}"

if [[ -n "${EM2MEM_LLM_PROFILE_OVERRIDE:-}" ]]; then
  _em2mem_llm_profile="${EM2MEM_LLM_PROFILE_OVERRIDE}"
elif [[ -f "${_em2mem_profile_file}" ]]; then
  _em2mem_llm_profile="$(tr -d '[:space:]' < "${_em2mem_profile_file}")"
elif [[ -n "${EM2MEM_LLM_PROFILE:-}" ]]; then
  _em2mem_llm_profile="${EM2MEM_LLM_PROFILE}"
else
  _em2mem_llm_profile="remote"
fi

case "${_em2mem_llm_profile}" in
  remote)
    export EM2MEM_LLM_PROFILE="remote"
    unset EM2MEM_LOCAL_LLM_ENABLED
    unset EM2MEM_LOCAL_LLM_NON_THINKING
    ;;
  local-qwen35)
    export EM2MEM_LLM_PROFILE="local-qwen35"
    export EM2MEM_LOCAL_LLM_ENABLED=1
    export EM2MEM_LOCAL_LLM_NON_THINKING="${EM2MEM_LOCAL_LLM_NON_THINKING:-1}"
    export EM2MEM_LOCAL_LLM_MAX_TOKENS="${EM2MEM_LOCAL_LLM_MAX_TOKENS:-4096}"
    export EM2MEM_LOCAL_LLM_CACHE_NAMESPACE="${EM2MEM_LOCAL_LLM_CACHE_NAMESPACE:-local_qwen35}"
    export EM2MEM_LOCAL_LLM_HOST="${EM2MEM_LOCAL_LLM_HOST:-127.0.0.1}"
    export EM2MEM_LOCAL_LLM_PORT="${EM2MEM_LOCAL_LLM_PORT:-18100}"
    export EM2MEM_LOCAL_LLM_BASE_URL="${EM2MEM_LOCAL_LLM_BASE_URL:-http://${EM2MEM_LOCAL_LLM_HOST}:${EM2MEM_LOCAL_LLM_PORT}/v1}"
    export EM2MEM_LOCAL_LLM_SERVED_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL:-gpt-5.4}"
    export OPENAI_BASE_URL="${EM2MEM_LOCAL_LLM_BASE_URL}"
    export OPENAI_API_KEY="${EM2MEM_LOCAL_LLM_API_KEY:-local-qwen35}"
    export OPENAI_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_MEMORY_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_QUERY_LLM_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_QUERY_RETRIEVER_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_QUERY_RESPOND_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_RESPOND_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_MCUR_ANSWER_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_MST_ANSWER_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_MST_REFINE_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_MST_EPISODIC_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_VLM_MODEL="${EM2MEM_LOCAL_LLM_SERVED_MODEL}"
    export EM2MEM_OPENAI_DISABLE_REASONING=1
    export EM2MEM_OPENAI_FORCE_CHAT_COMPLETIONS=1
    export EM2MEM_STREAM_ASR_CUDA_VISIBLE_DEVICES="${EM2MEM_LOCAL_STREAM_ASR_CUDA_VISIBLE_DEVICES:-1}"
    ;;
  *)
    echo "[llm_profile] unsupported EM2MEM_LLM_PROFILE=${_em2mem_llm_profile}; expected remote or local-qwen35" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

unset _em2mem_profile_root _em2mem_profile_file _em2mem_llm_profile
