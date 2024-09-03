# On GLM-4-flash
1. 即使前代无法正常执行，仍然可以演化出复杂代码
2. Syntax Error的反馈很不直观，模型难以通过正确地基于反馈修改代码
   - `Invalid syntex` 导致 `transform` 函数无法被找到时，反馈缺少该函数会导致模型在json外额外写一个函数
     - fix: 在prompt 中强调不要在json的code field外写代码，效果很好
   - `KeyError missing 'key'` 由于print(e)只会输出文字描述，不会输出Exception类型，print该error只会输出`key`
     - fix: 使用repr(e)代替str(e)，成功，但是仍然有json格式错误问题