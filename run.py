import sys
import json
import uuid
import typer
from pathlib import Path
from typing import Dict, Any
from src.groqService import GroqLLMWrapper

app = typer.Typer(help="CEIA NLP II TP-3 - CV Analysis Chatbot")

def extract_cv_info(cv_path: str, llm: GroqLLMWrapper) -> Dict[str, Any]:
    """
    Extract information from CV using LLM
    
    Args:
        cv_path (str): Path to the CV file
        llm: GroqLLMWrapper instance
    
    Returns:
        Dict containing name, lastname, profile_type, and person_id
    """
    try:
        # Read CV content
        with open(cv_path, 'r', encoding='utf-8') as file:
            cv_content = file.read()
        
        # Create prompt for LLM to extract information
        extraction_prompt = f"""
        Analiza el siguiente CV y extrae la siguiente información:
        
        - nombre: El nombre de la persona
        - apellido: El apellido de la persona  
        - tipo_perfil: Determina si es "desarrollador" o "soporte_tecnico" basándote en las habilidades y experiencia
        
        Responde con un objeto JSON que contenga estas claves: nombre, apellido, tipo_perfil
        
        CV a analizar:
        {cv_content[:500]}
        """
        
        # Send prompt to LLM with JSON mode
        response = llm.send_prompt_json(extraction_prompt)
        
        # Extract content from response
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content.strip()
            # typer.echo(f"LLM Response: {content}")
            
            # Parse JSON response (should be valid JSON due to json_object mode)
            cv_info = json.loads(content)
            
            # Generate unique ID
            person_id = str(uuid.uuid4())
            
            # Validate and return data
            return {
                "name": cv_info.get("nombre", "Unknown"),
                "lastname": cv_info.get("apellido", "Unknown"), 
                "profile_type": cv_info.get("tipo_perfil", "desarrollador"),
                "person_id": person_id
            }
                
    except json.JSONDecodeError as e:
        typer.echo(f"Warning: Could not parse LLM response for {cv_path}: {e}")
        typer.echo(f"Raw response: {content}")
    except Exception as e:
        typer.echo(f"Error extracting info from {cv_path}: {e}")
    
    # Fallback values for any error
    return {
        "name": "Unknown",
        "lastname": "Unknown",
        "profile_type": "desarrollador", 
        "person_id": str(uuid.uuid4())
    }

@app.command()
def ui(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server"),
    port: int = typer.Option(8050, help="Port to run the server"),
    debug: bool = typer.Option(True, help="Run in debug mode")
):
    """Launch the web UI interface"""
    try:
        from ui import app as dash_app
        typer.echo(f"Starting web UI at http://{host}:{port}")
        dash_app.run(debug=debug, host=host, port=port)
    except ImportError as e:
        typer.echo(f"Error importing UI: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error starting UI: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def chat():
    """Start CLI chat mode with the assistant"""
    try:
        from src.chatService import session
        
        typer.echo("Starting CLI chat mode...")
        typer.echo("Type 'exit' or 'quit' to end the conversation, or press Ctrl+C")
        typer.echo("=" * 50)
        
        while True:
            msg = input("You: ")
            if msg.lower() in ["exit", "quit"]:
                typer.echo("Goodbye!")
                break
            
            if msg.strip():  # Only process non-empty messages
                response = session.chat(msg)
                typer.echo(f"Assistant: {response}")
                print()  # For better readability
            
    except KeyboardInterrupt:
        typer.echo("\nNos vemos la proxima.")
        sys.exit(0)
    except ImportError as e:
        typer.echo(f"Error importing chat service: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error in chat mode: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def load_data(
    category: str = typer.Option("cv", help="Category for the CV data (default: 'cv')")
):
    """Load CV data into the vector database"""
    try:
        from src.vectorService import load_data_into_vectordb
        from src.config.settings import DATASET
        from src.groqService import GroqLLMWrapper
        
        typer.echo("Loading CV data into vector database...")
        
        llm = GroqLLMWrapper()     
        
        # Construct full paths for dataset files
        data_dir = Path("data")
        full_dataset_paths = []
        
        for cv_file in DATASET:
            full_path = data_dir / cv_file
            if full_path.exists():
                full_dataset_paths.append(str(full_path))
                typer.echo(f"Found: {full_path}")
            else:
                typer.echo(f"Missing: {full_path}")
                raise typer.Exit(1)
        
        typer.echo(f"Processing {len(full_dataset_paths)} CV files...")
        
        # Load data with progress indication
        with typer.progressbar(full_dataset_paths, label="Loading CVs") as progress:            
            for cv_path in progress:
                # Extract CV information using LLM
                cv_info = extract_cv_info(cv_path, llm)
                
                typer.echo(f"\nProcessing CV: {Path(cv_path).name}")
                typer.echo(f"  Name: {cv_info['name']} {cv_info['lastname']}")
                typer.echo(f"  Profile: {cv_info['profile_type']}")
                typer.echo(f"  ID: {cv_info['person_id']}")
                
                # Load data into vector database with extracted info
                load_data_into_vectordb(
                    [cv_path],
                    name=cv_info['name'],
                    lastname=cv_info['lastname'],
                    person_id=cv_info['person_id'],
                    profile_type=cv_info['profile_type'],
                    category=category,
                )
        
        typer.echo("Successfully loaded all CV data into vector database!")
        
    except ImportError as e:
        typer.echo(f"Error importing vector service: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error loading data: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query to find similar content"),
    top_k: int = typer.Option(5, help="Number of results to return"),
    debug: bool = typer.Option(True, help="Show detailed debug information")
):
    """Search for similar content in the vector database"""
    try:
        from src.vectorService import search_similar
        
        typer.echo(f"Searching for: '{query}'")
        typer.echo("=" * 50)
        
        results = search_similar(query, top_k=top_k, debug=debug)
        
        if not results:
            typer.echo("No results found.")
        else:
            typer.echo(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                typer.echo(f"{i}. {result}")
        
    except ImportError as e:
        typer.echo(f"Error importing vector service: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error during search: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()