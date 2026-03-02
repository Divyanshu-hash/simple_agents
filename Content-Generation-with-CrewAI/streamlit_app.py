from crewai import Agent, Task, Crew, LLM 
from crewai_tools import SerperDevTool
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="CrewAI - Content Generation", page_icon="🧠", layout="wide")

st.title("Content Generation with CrewAI")
st.markdown("Welcome to the CrewAI demo! This app demonstrates how CrewAI agents can collaborate to generate content efficiently.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Use text_input instead of text_area if multiline is not needed
    topic = st.text_input("Topic", "Medical Industry using Generative AI",
                          placeholder="Enter the topic for content generation")

    st.markdown(f"### LLM Configuration")
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)

    st.markdown(f"### Search Tool Configuration")
    n = st.slider("Number of Search Results", min_value=1, max_value=20, value=10, step=1)

    generate_button = st.button("Generate Content", type="primary")

# Information about the demo
with st.expander("Show Details"):
    st.markdown("""
        This demo showcases how CrewAI can be used to generate content collaboratively using AI agents.
        - **Senior Research Analyst:** Conducts research on the given topic.
        - **Content Writer:** Writes an article based on the research findings.
        - Tools include AI language models and web search capabilities.
    """)

# Function to generate content using CrewAI
def generate_content(topic):
    # Initialize AI model
    llm = LLM(model="groq/llama-3.3-70b-versatile")

    # Initialize search tool
    search_tool = SerperDevTool(n=n)

    # Define AI agents
    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal="Research on the Medical Industry using Generative AI",
        backstory="You are a Senior Research Analyst with 5 years of experience in AI and the medical industry.",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    content_writer = Agent(
        role="Content Writer",
        goal="Write a detailed article on the Medical Industry using Generative AI",
        backstory="You are an experienced Content Writer with expertise in AI and healthcare topics.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Define tasks for the agents
    research_task = Task(
        description=f"Research on {topic}",
        expected_output="A detailed summary of the current trends and applications of Generative AI in the Medical Industry",
        agent=senior_research_analyst
    )

    writing_task = Task(
        description=f"Write a detailed article on {topic}",
        expected_output="A well-researched article on the applications of Generative AI in the Medical Industry",
        agent=content_writer
    )

    # Assemble the crew
    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True
    )

    # Execute tasks and return result
    return crew.kickoff(inputs={"topic": topic})

# Trigger content generation on button click
if generate_button:
    with st.spinner("Generating content..."):
        try:
            result = generate_content(topic)
            st.markdown(f"### Content Generation Result")
            st.markdown(result)

            # Download option for the generated content
            st.download_button(
                label="Download Content",
                data=result.raw,
                file_name=f"{topic.replace(' ', '_')}_article.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer section
st.markdown("---")
st.markdown("Built with ❤️ by [CrewAI]")