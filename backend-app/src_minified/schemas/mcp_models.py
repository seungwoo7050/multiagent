H='1.0.0'
G=list
D=dict
C=True
A=None
import msgspec as B,time as E
from typing import Any,Dict,List,Optional,Union
from src.utils.ids import generate_uuid as F
class J(B.Struct,forbid_unknown_fields=C):role:0;content:0
class K(B.Struct,omit_defaults=C,forbid_unknown_fields=C):max_tokens=A;temperature=A;top_p=A;stop_sequences=A
class L(B.Struct,tag='llm_input',omit_defaults=C,forbid_unknown_fields=C):model:0;context_id=B.field(default_factory=lambda:F());timestamp=B.field(default_factory=E.time);metadata=B.field(default_factory=D);version=H;prompt=A;messages=A;parameters=A;use_cache=C;retry_on_failure=C
class M(B.Struct,forbid_unknown_fields=C):text=A;index=0;finish_reason=A
class N(B.Struct,omit_defaults=C,forbid_unknown_fields=C):prompt_tokens=A;completion_tokens=A;total_tokens=A
class O(B.Struct,tag='llm_output',omit_defaults=C,forbid_unknown_fields=C):success:0;context_id=B.field(default_factory=lambda:F());timestamp=B.field(default_factory=E.time);metadata=B.field(default_factory=D);version=H;result_text=A;choices=A;usage=A;error_message=A;model_used=A
class P(B.Struct,tag='conversation_turn',omit_defaults=C,forbid_unknown_fields=C):role:0;content:0;timestamp=B.field(default_factory=E.time);metadata=A
class I(B.Struct,forbid_unknown_fields=C):content:0;id=B.field(default_factory=F);parent_id=A;evaluation_score=A;status='generated';metadata=B.field(default_factory=D)
class Q(B.Struct,omit_defaults=C,forbid_unknown_fields=C):
	task_id:0;original_input:0;next_action=A;current_iteration=0;thoughts=B.field(default_factory=G);current_thoughts_to_evaluate=B.field(default_factory=G);current_best_thought_id=A;search_depth=0;max_search_depth=5;last_llm_input=A;last_llm_output=A;scratchpad='';tool_call_history=B.field(default_factory=G);next_node_override=A;final_answer=A;error_message=A;dynamic_data=B.field(default_factory=D);context_id=B.field(default_factory=F);timestamp=B.field(default_factory=E.time);metadata=B.field(default_factory=D);version=H
	def get_thought_by_id(B,thought_id):
		for A in B.thoughts:
			if A.id==thought_id:return A
	def add_thought(B,content,parent_id=A,metadata=A):A=I(content=content,parent_id=parent_id,metadata=metadata or{});B.thoughts.append(A);return A