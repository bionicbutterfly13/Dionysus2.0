"""
ASI-GO: Autonomous System Intelligence - General Optimizer
Main orchestrator for the problem-solving system, now with an integrated memory system.
"""
import os
import sys
import time
from colorama import init, Fore, Style
from dotenv import load_dotenv

# Import our modules
from llm_interface import LLMInterface
from memory_system import CognitionBase, EpisodicMemory, SemanticMemory
from researcher import Researcher
from engineer import Engineer
from analyst import Analyst
from utils import setup_logging, save_checkpoint, print_header, print_step

# Initialize colorama
init(autoreset=True)

class ASIGO:
    """Main orchestrator for the ASI-GO system."""

    def __init__(self):
        """Initialize all components, including the memory system."""
        print_header("ASI-GO System Initialization")
        self.logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))
        self.logger.info("Starting ASI-GO system")

        try:
            # 1. Initialize Memory System
            print_step("1/5", "Initializing Memory System...")
            self.procedural_memory = CognitionBase() # Procedural
            self.episodic_memory = EpisodicMemory()   # Episodic
            self.semantic_memory = SemanticMemory() # Semantic
            print(f"   {Fore.GREEN}✓ Memory system loaded")

            # 2. Initialize LLM Interface
            print_step("2/5", "Initializing LLM Interface...")
            self.llm = LLMInterface()
            provider_info = self.llm.get_provider_info()
            print(f"   {Fore.GREEN}✓ Using {provider_info['provider']} - {provider_info['model']}")

            # 3. Initialize Core Components
            print_step("3/5", "Initializing Researcher...")
            self.researcher = Researcher(self.llm, self.procedural_memory)
            print(f"   {Fore.GREEN}✓ Researcher ready")

            print_step("4/5", "Initializing Engineer...")
            self.engineer = Engineer()
            print(f"   {Fore.GREEN}✓ Engineer ready")

            print_step("5/5", "Initializing Analyst...")
            self.analyst = Analyst(self.llm, self.procedural_memory, self.semantic_memory)
            print(f"   {Fore.GREEN}✓ Analyst ready")

            self.max_iterations = int(os.getenv("MAX_ITERATIONS", "5"))
            print(f"\n{Fore.GREEN}✓ System initialized successfully!")

        except Exception as e:
            print(f"\n{Fore.RED}✗ Initialization failed: {e}")
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            sys.exit(1)

    def solve_problem(self, goal: str) -> bool:
        """Main problem-solving loop, now with episodic memory recording."""
        print_header(f"Solving: {goal}")
        self.episodic_memory.start_episode(goal)
        
        iteration = 0
        previous_attempt = None
        success = False

        while iteration < self.max_iterations and not success:
            iteration += 1
            print(f"\n{Fore.CYAN}{'='*60}")
            print(f"{Fore.CYAN}Iteration {iteration}/{self.max_iterations}")
            print(f"{Fore.CYAN}{'='*60}\n")

            try:
                # 1. Researcher proposes a solution
                print_step("Research", "Generating solution proposal...")
                proposal, workspace = self.researcher.propose_solution(goal, previous_attempt)
                self.episodic_memory.record_step("Research", proposal)
                print(f"   {Fore.GREEN}✓ Proposal generated (Pattern: {proposal.get('pattern_used')})")

                # 2. Engineer tests the solution
                print_step("Engineering", "Testing proposed solution...")
                test_result = self.engineer.test_solution(proposal)
                self.episodic_memory.record_step("Engineering", test_result)
                if test_result['success']:
                    print(f"   {Fore.GREEN}✓ Execution successful")
                else:
                    print(f"   {Fore.RED}✗ Execution failed: {test_result.get('error', 'Unknown')}")

                # 3. Analyst analyzes the results
                print_step("Analysis", "Analyzing results...")
                validation = self.engineer.validate_output(test_result.get('output', ''), goal)
                analysis = self.analyst.analyze_results(proposal, test_result, validation, workspace)
                self.episodic_memory.record_step("Analysis", analysis)
                print(f"   {Fore.GREEN}✓ Analysis complete")

                # Check for success
                if test_result['success'] and validation.get('meets_goal', False):
                    success = True
                    print(f"\n{Fore.GREEN}✓ Goal achieved successfully!")
                else:
                    previous_attempt = test_result
                    print(f"\n   {Fore.YELLOW}Recommendation: {self.analyst.recommend_next_action()}")

            except Exception as e:
                self.logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)
                print(f"\n{Fore.RED}✗ An unexpected error occurred in iteration {iteration}.")
                previous_attempt = {"error": str(e)}

        self.episodic_memory.end_episode(success)
        print_header("Final Report")
        print(self.analyst.generate_summary_report())
        return success

    def interactive_mode(self):
        """Run in interactive mode."""
        print_header("ASI-GO Interactive Mode")
        print("Enter your problem-solving goals. Type 'exit' to quit.\n")
        while True:
            try:
                goal = input(f"{Fore.CYAN}Enter your goal: {Style.RESET_ALL}").strip()
                if goal.lower() in ['exit', 'quit', 'q']:
                    break
                if not goal:
                    continue
                self.solve_problem(goal)
                if input(f"\n{Fore.CYAN}Solve another problem? (y/n): {Style.RESET_ALL}").lower() != 'y':
                    break
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                self.logger.error(f"Interactive mode error: {e}", exc_info=True)
                print(f"\n{Fore.RED}An unexpected error occurred.")

        print(f"\n{Fore.YELLOW}Thank you for using ASI-GO!")

def main():
    load_dotenv()
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}{'ASI-GO: Autonomous System Intelligence'.center(60)}")
    print(f"{Fore.CYAN}{'v2.0 with Integrated Memory'.center(60)}")
    print(f"{Fore.CYAN}{'='*60}\n")
    asi_go = ASIGO()
    if len(sys.argv) > 1:
        asi_go.solve_problem(' '.join(sys.argv[1:]))
    else:
        asi_go.interactive_mode()

if __name__ == "__main__":
    main()
