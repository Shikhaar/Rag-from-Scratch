"""
Interactive CLI and REPL Interface for RAG From First Principles.
"""

import argparse
import os
import sys
import time
from typing import Optional

from rag_pipeline import RAGPipeline
from evaluation.dataset import EvalDataset
from evaluation.benchmark import RAGBenchmark, print_benchmark_report
from core.llm import GeminiLLM, OllamaLLM, MockLLM

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.text import Text
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def print_banner():
    title = """
===================================================================
                RAG FROM FIRST PRINCIPLES
      Dense Vector Search | BM25 Inverted Index | RRF | Reranker
===================================================================
    """
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]RAG FROM FIRST PRINCIPLES[/bold cyan]\n"
            "[dim]A production-oriented Retrieval-Augmented Generation system built from scratch in Python[/dim]\n"
            "[green]NumPy Dense Store | Okapi BM25 Index | Reciprocal Rank Fusion | Cross-Encoder Reranker[/green]",
            border_style="cyan"
        ))
    else:
        print(title)


def display_inspect_results(query: str, results):
    """Render granular score breakdown across all retrieval stages."""
    if HAS_RICH:
        console.print(f"\n[bold yellow]Query:[/bold yellow] [bold white]\"{query}\"[/bold white]")
        console.print("-" * 70)

        for rank, res in enumerate(results, 1):
            source = res.metadata.get("filename", res.metadata.get("source", "Unknown"))
            page = res.metadata.get("page", "N/A")
            chunk_idx = res.chunk.chunk_index
            
            d_score = f"{res.dense_score:.4f}" if res.dense_score is not None else "N/A"
            b_score = f"{res.bm25_score:.4f}" if res.bm25_score is not None else "N/A"
            r_score = f"{res.rrf_score:.5f}" if res.rrf_score is not None else "N/A"
            re_score = f"{res.rerank_score:.4f}" if res.rerank_score is not None else "N/A"

            panel_content = (
                f"[bold cyan]{rank}. {source}[/bold cyan]  (page: {page}, chunk: {chunk_idx}, id: {res.chunk.chunk_id})\n\n"
                f"   [bold green]Dense Score:[/bold green]  {d_score:<10} "
                f"[bold blue]BM25 Score:[/bold blue]  {b_score:<10}\n"
                f"   [bold magenta]RRF Score:[/bold magenta]    {r_score:<10} "
                f"[bold red]Rerank Score:[/bold red] {re_score:<10}\n\n"
                f"[dim]\"{res.chunk.content.strip()[:240]}...\"[/dim]"
            )
            console.print(Panel(panel_content, border_style="blue" if rank == 1 else "dim"))
    else:
        print(f"\nQuery:\n\"{query}\"\n" + "-" * 60)
        for rank, res in enumerate(results, 1):
            source = res.metadata.get("filename", "Unknown")
            page = res.metadata.get("page", "N/A")
            print(f"\n{rank}. {source} (page: {page}, chunk: {res.chunk.chunk_index})")
            print(f"   Dense Score:  {res.dense_score}")
            print(f"   BM25 Score:   {res.bm25_score}")
            print(f"   RRF Score:    {res.rrf_score}")
            print(f"   Rerank Score: {res.rerank_score}")
            print(f"   \"{res.chunk.content.strip()[:200]}...\"\n")
            print("-" * 60)


def display_query_result(res):
    """Render answer, citations, and latency."""
    if HAS_RICH:
        console.print("\n[bold green]=== Grounded Answer ===[/bold green]")
        console.print(Markdown(res.answer))
        
        if res.citations:
            console.print("\n[bold cyan]=== Sources & Citations ===[/bold cyan]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Tag", width=12)
            table.add_column("Source Document", width=26)
            table.add_column("Chunk", width=8)
            table.add_column("Preview Snippet")

            for c in res.citations:
                table.add_row(
                    f"[bold yellow]{c['label']}[/bold yellow]",
                    str(c['source']),
                    str(c['chunk_index']),
                    c['preview'][:100] + "..."
                )
            console.print(table)

        console.print(f"[dim]Latency: {res.latency_ms:.1f}ms | Retrieval Mode: {res.metadata.get('retrieval_mode')}[/dim]\n")
    else:
        print("\n=== Grounded Answer ===")
        print(res.answer)
        if res.citations:
            print("\n=== Sources & Citations ===")
            for c in res.citations:
                print(f"  {c['label']} -> {c['source']} (chunk {c['chunk_index']})")
                print(f"     \"{c['preview'][:100]}...\"")
        print(f"\n[Latency: {res.latency_ms:.1f}ms]\n")


def run_repl(pipeline: RAGPipeline):
    """Start interactive command loop."""
    print_banner()
    if HAS_RICH:
        console.print("[dim]Type your question directly, or use /help for available commands.[/dim]\n")
    else:
        print("Type your question directly, or use /help for available commands.\n")

    current_mode = "hybrid"

    while True:
        try:
            prompt_str = f"rag [{current_mode}]> "
            user_input = input(prompt_str).strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("Goodbye!")
                break

            elif user_input.startswith("/help"):
                print("""
Available Commands:
  /query <question>       - Run full RAG pipeline and generate grounded response
  /inspect <question>     - Inspect multi-stage retrieval scores (Dense, BM25, RRF, Rerank)
  /ingest <path>          - Ingest a file (.txt, .md, .pdf) or entire directory
  /benchmark              - Run 4-way evaluation benchmark (BM25 vs Dense vs Hybrid vs Reranker)
  /mode <hybrid|dense|bm25> - Switch active retrieval strategy
  /stats                  - View indexing and vector store statistics
  /clear                  - Clear terminal screen
  /exit                   - Exit CLI
                """)

            elif user_input.startswith("/clear"):
                os.system("cls" if os.name == "nt" else "clear")
                print_banner()

            elif user_input.startswith("/mode"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1] in ["hybrid", "dense", "sparse", "bm25"]:
                    current_mode = parts[1]
                    print(f"Retrieval mode switched to: {current_mode}")
                else:
                    print("Usage: /mode <hybrid|dense|bm25>")

            elif user_input.startswith("/stats"):
                stats = pipeline.stats()
                print("\n=== System Statistics ===")
                for k, v in stats.items():
                    print(f"  {k:<24}: {v}")
                print()

            elif user_input.startswith("/ingest"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /ingest <path-to-file-or-dir>")
                    continue
                path = parts[1].strip("\"'")
                if not os.path.exists(path):
                    print(f"Error: Path '{path}' does not exist.")
                    continue
                
                print(f"Ingesting: {path} ...")
                if os.path.isdir(path):
                    res = pipeline.ingest_directory(path)
                else:
                    res = pipeline.ingest_file(path)
                print(f"Ingestion complete: {res}")

            elif user_input.startswith("/inspect"):
                query_text = user_input[len("/inspect"):].strip()
                if not query_text:
                    print("Usage: /inspect <question>")
                    continue
                results = pipeline.inspect(query_text, top_k=5)
                display_inspect_results(query_text, results)

            elif user_input.startswith("/benchmark"):
                eval_path = os.path.join(os.path.dirname(__file__), "data", "eval_dataset.json")
                if not os.path.exists(eval_path):
                    print(f"Evaluation dataset not found at {eval_path}")
                    continue
                print("Running automated 4-way evaluation benchmark across corpus...")
                ds = EvalDataset.load(eval_path)
                benchmark = RAGBenchmark(
                    dense_retriever=pipeline.dense_retriever,
                    sparse_retriever=pipeline.sparse_retriever,
                    reranker=pipeline.reranker,
                    dataset=ds,
                )
                results = benchmark.run_benchmark(all_chunks=pipeline.chunks, k=5)
                print_benchmark_report(results, k=5)

            elif user_input.startswith("/query"):
                query_text = user_input[len("/query"):].strip()
                if not query_text:
                    print("Usage: /query <question>")
                    continue
                res = pipeline.query(query_text, mode=current_mode)
                display_query_result(res)

            else:
                # Default behavior: run query
                res = pipeline.query(user_input, mode=current_mode)
                display_query_result(res)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG From First Principles CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    p_ingest = subparsers.add_parser("ingest", help="Ingest a file or directory")
    p_ingest.add_argument("path", help="Path to file or folder")

    # Query command
    p_query = subparsers.add_parser("query", help="Query the RAG pipeline")
    p_query.add_argument("question", help="Question string")
    p_query.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25", "sparse"])

    # Inspect command
    p_inspect = subparsers.add_parser("inspect", help="Inspect multi-stage scores for a question")
    p_inspect.add_argument("question", help="Question string")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Run retrieval benchmark")

    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive REPL")

    args = parser.parse_args()

    # Determine LLM provider (Gemini if API key present, otherwise MockLLM)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        llm = GeminiLLM(api_key=gemini_key)
    else:
        llm = MockLLM()

    pipeline = RAGPipeline(llm=llm)

    # Ingest default sample docs if store is empty
    if len(pipeline.chunks) == 0:
        sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_knowledge")
        if os.path.exists(sample_dir):
            pipeline.ingest_directory(sample_dir)

    if args.command == "ingest":
        if os.path.isdir(args.path):
            res = pipeline.ingest_directory(args.path)
        else:
            res = pipeline.ingest_file(args.path)
        print(f"Ingestion result: {res}")

    elif args.command == "query":
        res = pipeline.query(args.question, mode=args.mode)
        display_query_result(res)

    elif args.command == "inspect":
        results = pipeline.inspect(args.question, top_k=5)
        display_inspect_results(args.question, results)

    elif args.command == "benchmark":
        eval_path = os.path.join(os.path.dirname(__file__), "data", "eval_dataset.json")
        ds = EvalDataset.load(eval_path)
        benchmark = RAGBenchmark(
            dense_retriever=pipeline.dense_retriever,
            sparse_retriever=pipeline.sparse_retriever,
            reranker=pipeline.reranker,
            dataset=ds,
        )
        results = benchmark.run_benchmark(all_chunks=pipeline.chunks, k=5)
        print_benchmark_report(results, k=5)

    else:
        # Default to interactive REPL
        run_repl(pipeline)


if __name__ == "__main__":
    main()
