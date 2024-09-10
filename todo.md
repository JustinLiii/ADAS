- [ ] LLM Generated code Sanbox
  - [ ] OOM Problem
  - [ ] Dead Loop (Timeout)
  - [ ] Console output
  - [ ] Errors
  - [ ] System functions
- [ ] API stop responding (May be because of LLM timeout instead of dead loop?)
- [ ] Redesign get_response:
  - [ ] Can we merge `get_json_response_from_gpt` and `get_json_response_from_gpt_reflect` ?
  - [ ] Better reflex formatting error to next round
- [ ] Batch API
- [X] Swith Model (To GPT-4o-mini + GLM-4-flash)
- [ ] Outside configurable LLMAgentBase
- [ ] Faster evaluation
- [X] Detail statistic on experiment token cost
- [ ] Get a GPT-4 + GPT-3.5 generation log for reference

- [X] JSON experiment setting
  - [X] Saving
- [X] Design log
  - [x] next_solution reference in `finally` but not assigned in `try`
  - [X] exception output in `threadpool` in `evaluate_forward_fn`
  - [X] Use Logger
- [X] Better Experiment Naming


- 用glm4 plus做agent模型的准确率是13%上下