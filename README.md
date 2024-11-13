# **Lab Exercise 2: Optimizing the AI-Generated Code with Few-Shot Prompt Technique**

**Purpose**  
This lab focuses on using few-shot prompt techniques with the IBM Granite Code model on Google Colab to improve code generation quality for Jupyter Notebook `ipywidgets`-based UI components. Few-shot prompting allows the model to draw from multiple examples, enabling more accurate and nuanced outputs than the initial zero-shot prompt approach.

**Lab Overview**  
Building upon the previous zero-shot prompting lab, this exercise uses the few-shot technique to optimize the code generated for an online bookstore’s UI components in Jupyter Notebook. Through a structured process, you'll create and refine examples to improve the AI's code generation quality.

## Prerequisites
- Completion of Lab 1: **Code Generation using Zero-Shot Prompt**.
- Familiarity with Python and Jupyter Notebook `ipywidgets`.
- **Replicate Account**: An account with Replicate for model inference calls.
- **REPLICATE_API_TOKEN**: A Replicate API token stored as a Google Colab secret.

---

## Setting Up Google Colab for Few-Shot Prompting

### Step 1: Set Up a Replicate Account (If Not Done Already)
1. Visit [Replicate](https://replicate.com/) and create an account.
2. Retrieve your API token from Replicate.

### Step 2: Store REPLICATE_API_TOKEN and Set Up the Colab Environment
1. Open Google Colab, navigate to "Secrets" on the left-hand panel.
2. Add a new secret with the name `REPLICATE_API_TOKEN` and paste your Replicate API token.

---

## Optimizing Code Generation with Few-Shot Prompt Technique

### Step 1: Set Up the Model and Install Necessary Libraries

In this lab, you will reuse the setup from Lab 1, initializing the IBM Granite Code model in Google Colab. Follow these steps:

#### Cell 1: Install required packages

```bash
# Install necessary packages
!pip install git+https://github.com/ibm-granite-community/utils \
            "langchain_community<0.3.0" \
            replicate
```

#### Cell 2: Initialize the Replicate model
```python
# Initilize the Replicate & Colab Libraries

from ibm_granite_community.notebook_utils import get_env_var
from langchain_community.llms import Replicate
 
# Configure and load the IBM Granite model for code generation
model = Replicate(
    model="ibm-granite/granite-8b-code-instruct-128k",
    replicate_api_token=get_env_var('REPLICATE_API_TOKEN'),
    model_kwargs={"max_length": 100, "temperature": 0.2},
)
```

---

### Step 2: Define Few-Shot Prompt Method and Generate Enhanced UI Code

In this step, you'll define a few-shot prompt by giving the model example questions and contexts, guiding it toward creating more customized UI components.

#### Cell 1: Define the Few-Shot Prompt Examples

The few-shot examples below demonstrate specific tasks, such as creating a catalog tile UI with styling, adding a search bar, and designing an alternating tile color layout.

```python
examples = [
    {
        "question": "Create a UI for my landing page showing a catalog for my online bookstore",
        "context": "Simple Catalog Tile with Borders and Shadows",
        "output": """
import ipywidgets as widgets
from IPython.display import display

# Create a container for the book catalog
container = widgets.VBox()

# Create styled book tiles
book_tiles = [
    widgets.HTML(value="
        <div style='background-color: #f5f5f5; padding: 15px; margin: 10px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);'>
            <h2 style='color: #333;'>Book Title 1</h2>
            <p style='color: #333;'><b>Author:</b> Author Name 1</p>
            <p style='color: #333;'><b>Price:</b> $10.00</p>
        </div>"),
    widgets.HTML(value="
        <div style='background-color: #f5f5f5; padding: 15px; margin: 10px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);'>
            <h2 style='color: #333;'>Book Title 2</h2>
            <p style='color: #333;'><b>Author:</b> Author Name 2</p>
            <p style='color: #333;'><b>Price:</b> $15.00</p>
        </div>"),
]

container.children = book_tiles
display(container)
"""
    },
    {
        "question": "Add a search bar to my online bookstore's landing page with a placeholder and search icon",
        "context": "Search Bar with Placeholder Text and Search Button",
        "output": """
import ipywidgets as widgets
from IPython.display import display

# Create search bar with placeholder text and search icon
search_bar = widgets.HBox([
    widgets.Text(
        placeholder="Search for a book...",
        layout=widgets.Layout(width="250px", padding="5px"),
        style={"description_width": "0px"}
    ),
    widgets.Button(
        icon="search",
        tooltip="Search",
        layout=widgets.Layout(width="40px", padding="5px")
    )
])

display(search_bar)
"""
    },
    {
        "question": "Design an enhanced catalog UI with alternating tile colors for my online bookstore",
        "context": "Enhanced Book Catalog with Alternating Tile Colors and Font Styling",
        "output": """
import ipywidgets as widgets
from IPython.display import display

# Create search bar with placeholder text and search icon
search_bar = widgets.HBox([
    widgets.Text(
        placeholder="Search for a book...",
        layout=widgets.Layout(width="250px", padding="5px"),
        style={"description_width": "0px"}
    ),
    widgets.Button(
        icon="search",
        tooltip="Search",
        layout=widgets.Layout(width="40px", padding="5px")
    )
])

display(search_bar)
"""
    }
]
```
#### Cell 2: Define the Few-Shot Prompt Examples
Use the examples defined above to guide the model in producing an optimized UI layout with multiple components, including a catalog, search bar, and enhanced tile styling.
```python
def fewshot_prompt(examples):
    """
    Formats and combines few-shot examples for the model.
    
    Parameters:
    - examples: list of example dictionaries with question, context, and output
    
    Returns:
    - str, a combined prompt for the model
    """
    formatted_examples = "\n".join(
        f"Question: {example['question']}\nContext: {example['context']}\nOutput:\n{example['output']}"
        for example in examples
    )
    prompt = f"You are an experienced developer building a UI for an online bookstore. Here are some examples:\n\n{formatted_examples}"
    return prompt
```

# Invoke model with few-shot prompt
fewshot_code = fewshot_prompt(examples)
print("Generated Few-Shot Code:\n", fewshot_code)


### Step 3: Validate the Output

In this step, validate the few-shot generated code by running it in your Jupyter Notebook. This code should create a catalog with styled book tiles, a search bar with an icon, and alternating colors for improved readability.

```python
import ipywidgets as widgets
from IPython.display import display

# Create a container for the book catalog
container = widgets.VBox()

# Create styled book tiles
book_tiles = [
    widgets.HTML(value="""
        <div style='background-color: #f5f5f5; padding: 15px; margin: 10px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);'>
            <h2 style='color: #333;'>Book Title 1</h2>
            <p style='color: #333;'><b>Author:</b> Author Name 1</p>
            <p style='color: #333;'><b>Price:</b> $10.00</p>
        </div>"""),
    widgets.HTML(value="""
        <div style='background-color: #f5f5f5; padding: 15px; margin: 10px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);'>
            <h2 style='color: #333;'>Book Title 2</h2>
            <p style='color: #333;'><b>Author:</b> Author Name 2</p>
            <p style='color: #333;'><b>Price:</b> $15.00</p>
        </div>"""),
]

container.children = book_tiles

# Create search bar with placeholder text and search icon
search_bar = widgets.HBox([
    widgets.Text(
        placeholder="Search for a book...",
        layout=widgets.Layout(width="250px", padding="5px"),
        style={"description_width": "0px"}
    ),
    widgets.Button(
        icon="search",
        tooltip="Search",
        layout=widgets.Layout(width="40px", padding="5px")
    )
])

# Display the landing page
display(widgets.VBox([container, search_bar]))