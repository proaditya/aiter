"""render_chk.py: gpt-oss chat prompt token ids as vLLM's chat endpoint builds them (system message with reasoning
effort low and a fixed date, then the user message; render for completion), for the first questions of the MMLU-Pro
sample. Run in each vLLM image (no GPU needed) and compare the printed digests:
  docker run --rm --entrypoint python3 -v <dir with this script and the sample>:/e <vllm image> /e/render_chk.py
      [/e/mmlupro_test.jsonl] [date]"""
import hashlib
import json
import string

import sys
from openai_harmony import (Conversation, HarmonyEncodingName, Message, ReasoningEffort, Role, SystemContent,
                            load_harmony_encoding)

try:
    from importlib.metadata import version
    print("openai-harmony", version("openai-harmony"))
except Exception as e:  # noqa: BLE001
    print("openai-harmony version unknown", e)
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
TEMPLATE = ("Answer the following multiple choice question. The last line of your response should be of the following "
            "format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before "
            "answering.\n\n{question}\n\n{options}")
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "/e/mmlupro_test.jsonl"
DATE = sys.argv[2] if len(sys.argv) > 2 else "2026-10-18"
recs = [json.loads(l) for l in open(SAMPLE)][:5]
allh = hashlib.md5()
for r in recs:
    L = string.ascii_uppercase[: len(r["options"])]
    msg = TEMPLATE.format(letters=L, question=r["question"],
                          options="\n".join("%s) %s" % (x, o) for x, o in zip(L, r["options"])))
    sysc = SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW).with_conversation_start_date(DATE)
    conv = Conversation.from_messages([Message.from_role_and_content(Role.SYSTEM, sysc),
                                       Message.from_role_and_content(Role.USER, msg)])
    ids = enc.render_conversation_for_completion(conv, Role.ASSISTANT)
    h = hashlib.md5(json.dumps(ids).encode()).hexdigest()[:12]
    allh.update(h.encode())
    print("id %s: %d tokens, md5 %s" % (r["id"], len(ids), h))
print("ALL", allh.hexdigest()[:12])
print("system text:", enc.decode(enc.render_conversation_for_completion(
    Conversation.from_messages([Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(
        ReasoningEffort.LOW).with_conversation_start_date(DATE))]), Role.ASSISTANT)).replace("\n", " | "))
