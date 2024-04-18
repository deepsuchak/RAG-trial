from llama_index.core import PromptTemplate


instruction_str = """\ 
    1. The code should represent a solution to the query.
    2. PRINT ONLY THE EXPRESSION.
    3. Do not quote the expression."""

new_prompt = PromptTemplate(
    """\
    You are working with a pdf file in python.
    The name of the pdf file is `data-space.pdf`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)

context = """Purpose: The primary role of this agent is to assist users by providing accurate 
            information about the Space, Blackholes, Universe, Galaxies, Terraforming and it's Mystery. """