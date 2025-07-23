class SelfCorrectingTradeParser:
 
    def __init__(self, llm, max_iter: int = 3):
        # LLM object such as LangChain's ChatNVIDIA
        self.llm = llm
        # Maximum number of self-correction iterations to try
        self.max_iter = max_iter
 
    def parse_with_self_correction(self, trade_text: str) -> dict:
        # Create prompt describing the trade parsing process
        prompt = self._initial_prompt(trade_text)
        for _ in range(self.max_iter):
            # Get the LLM response
            reply = self.llm(prompt)
            # Extract trade dictionary and string template
            trade_dict, tmpl = self._extract(reply)
            # See if we can reconstruct the original trade text
            diffs = self._validate(trade_text, tmpl, trade_dict):
            # If there are no differences/errors, we are done
            if not diffs:
                return trade_dict
            # Otherwise we create a prompt with the corrections and retry
            prompt = self._correction_prompt(trade_text, trade_dict, tmpl, diffs)
 
        return trade_dict
