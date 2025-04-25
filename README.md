# classification
# Data Analysis Agent

A sophisticated data analysis agent that can perform statistical analysis, generate reports, and explain metrics.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from data_analysis_agent import DataAnalysisAgent, create_analysis_agent
from langchain_openai import ChatOpenAI

# Initialize with an LLM
llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
agent = create_analysis_agent(llm)

# Load data
import pandas as pd
df = pd.read_csv("your_data.csv")
agent.load_dataset(df)

# Generate report
report = agent.generate_report("classification analysis")

# Save the agent
agent.save_pretrained("saved_agent")

# Later load the agent
loaded_agent = DataAnalysisAgent.from_pretrained("saved_agent", llm=llm)
```

## Features

- Statistical analysis
- Metric explanations
- Visualization generation
- Conversation history
- Hugging Face Hub integration

## License

MIT
