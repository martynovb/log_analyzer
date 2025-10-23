"""
Simple UI Module for Log Analyzer

This module provides a simple GUI interface for uploading logs
and entering issue descriptions.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional, Callable
import os


class LogAnalyzerUI:
    """Simple GUI interface for the log analyzer."""
    
    def __init__(self, on_analyze_callback: Optional[Callable] = None):
        """
        Initialize the UI.
        
        Args:
            on_analyze_callback: Callback function called when analyze button is clicked
        """
        self.on_analyze_callback = on_analyze_callback
        self.log_file_path = ""
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Log Analyzer")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create and layout all UI widgets."""
        
        # Title
        title_label = ttk.Label(self.main_frame, text="Log Analyzer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Log file selection
        ttk.Label(self.main_frame, text="Log File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        file_frame = ttk.Frame(self.main_frame)
        file_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(0, weight=1)
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly")
        self.file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.browse_button = ttk.Button(file_frame, text="Browse", command=self._browse_file)
        self.browse_button.grid(row=0, column=1)
        
        # Issue description
        ttk.Label(self.main_frame, text="Issue Description:").grid(row=2, column=0, sticky=(tk.W, tk.N), pady=5)
        
        self.issue_text = scrolledtext.ScrolledText(self.main_frame, height=10, width=50)
        self.issue_text.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Placeholder text
        self.issue_text.insert("1.0", "Describe the issue you're experiencing...")
        self.issue_text.bind("<FocusIn>", self._on_text_focus_in)
        self.issue_text.bind("<FocusOut>", self._on_text_focus_out)
        
        # Analysis options frame
        options_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        options_frame.columnconfigure(1, weight=1)
        
        # Date range
        ttk.Label(options_frame, text="Date Range:").grid(row=0, column=0, sticky=tk.W, pady=2)
        date_frame = ttk.Frame(options_frame)
        date_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        self.start_date_var = tk.StringVar()
        self.end_date_var = tk.StringVar()
        
        ttk.Label(date_frame, text="From:").grid(row=0, column=0, padx=(0, 5))
        start_date_entry = ttk.Entry(date_frame, textvariable=self.start_date_var, width=12)
        start_date_entry.grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(date_frame, text="To:").grid(row=0, column=2, padx=(0, 5))
        end_date_entry = ttk.Entry(date_frame, textvariable=self.end_date_var, width=12)
        end_date_entry.grid(row=0, column=3)
        
        # Analysis type
        ttk.Label(options_frame, text="Analysis Type:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.analysis_type_var = tk.StringVar(value="analysis")
        analysis_combo = ttk.Combobox(options_frame, textvariable=self.analysis_type_var, 
                                    values=["analysis", "debugging", "root_cause", "solution"], 
                                    state="readonly", width=15)
        analysis_combo.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Max tokens
        ttk.Label(options_frame, text="Max Tokens:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.max_tokens_var = tk.StringVar(value="3000")
        tokens_entry = ttk.Entry(options_frame, textvariable=self.max_tokens_var, width=10)
        tokens_entry.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.main_frame)
        buttons_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        self.analyze_button = ttk.Button(buttons_frame, text="Analyze Logs", 
                                       command=self._on_analyze_click, style="Accent.TButton")
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(buttons_frame, text="Clear", command=self._clear_form)
        self.clear_button.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, 
                                          mode='indeterminate')
        self.progress_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Results area
        results_frame = ttk.LabelFrame(self.main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8, width=70)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main frame row weights
        self.main_frame.rowconfigure(6, weight=1)
    
    def _browse_file(self):
        """Open file dialog to select log file."""
        file_path = filedialog.askopenfilename(
            title="Select Log File",
            filetypes=[
                ("Log files", "*.log"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.log_file_path = file_path
            self.file_path_var.set(os.path.basename(file_path))
    
    def _on_text_focus_in(self, event):
        """Handle text widget focus in event."""
        if self.issue_text.get("1.0", "end-1c") == "Describe the issue you're experiencing...":
            self.issue_text.delete("1.0", "end-1c")
            self.issue_text.config(foreground="black")
    
    def _on_text_focus_out(self, event):
        """Handle text widget focus out event."""
        if not self.issue_text.get("1.0", "end-1c").strip():
            self.issue_text.insert("1.0", "Describe the issue you're experiencing...")
            self.issue_text.config(foreground="gray")
    
    def _on_analyze_click(self):
        """Handle analyze button click."""
        # Validate inputs
        if not self.log_file_path:
            messagebox.showerror("Error", "Please select a log file.")
            return
        
        issue_description = self.issue_text.get("1.0", "end-1c").strip()
        if not issue_description or issue_description == "Describe the issue you're experiencing...":
            messagebox.showerror("Error", "Please enter an issue description.")
            return
        
        try:
            max_tokens = int(self.max_tokens_var.get())
            if max_tokens <= 0:
                raise ValueError("Max tokens must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for max tokens.")
            return
        
        # Show progress
        self.progress_bar.start()
        self.analyze_button.config(state="disabled")
        
        # Call callback if provided
        if self.on_analyze_callback:
            try:
                self.on_analyze_callback(self._get_analysis_data())
            except Exception as e:
                messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            finally:
                self.progress_bar.stop()
                self.analyze_button.config(state="normal")
    
    def _clear_form(self):
        """Clear all form fields."""
        self.log_file_path = ""
        self.file_path_var.set("")
        self.issue_text.delete("1.0", "end-1c")
        self.issue_text.insert("1.0", "Describe the issue you're experiencing...")
        self.issue_text.config(foreground="gray")
        self.start_date_var.set("")
        self.end_date_var.set("")
        self.analysis_type_var.set("analysis")
        self.max_tokens_var.set("3000")
        self.results_text.delete("1.0", "end-1c")
    
    def _get_analysis_data(self) -> dict:
        """Get analysis data from form."""
        issue_description = self.issue_text.get("1.0", "end-1c").strip()
        if issue_description == "Describe the issue you're experiencing...":
            issue_description = ""
        
        return {
            "log_file_path": self.log_file_path,
            "issue_description": issue_description,
            "start_date": self.start_date_var.get().strip() or None,
            "end_date": self.end_date_var.get().strip() or None,
            "analysis_type": self.analysis_type_var.get(),
            "max_tokens": int(self.max_tokens_var.get())
        }
    
    def show_results(self, results: str):
        """Display analysis results in the results area."""
        self.results_text.delete("1.0", "end-1c")
        self.results_text.insert("1.0", results)
    
    def show_error(self, error_message: str):
        """Display error message."""
        messagebox.showerror("Error", error_message)
    
    def show_info(self, message: str):
        """Display info message."""
        messagebox.showinfo("Info", message)
    
    def run(self):
        """Start the UI main loop."""
        self.root.mainloop()
    
    def destroy(self):
        """Destroy the UI."""
        self.root.destroy()
