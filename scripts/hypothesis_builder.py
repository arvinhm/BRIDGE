#!/usr/bin/env python3
"""
================================================================================
HYPOTHESIS PATHWAY BUILDER - Interactive GUI
================================================================================

A beautiful, interactive GUI for building TF→Gene regulatory hypotheses.

Features:
- Drag-and-drop TF (squares) and Gene (circles) nodes
- Connect nodes with activating (→) or repressing (⊣) arrows
- Double-click to name nodes
- Right-click to delete connections
- Export hypothesis for testing with the GRN model
- Beautiful modern design

Usage:
    python scripts/hypothesis_builder.py
    
    # Or with predefined structure:
    python scripts/hypothesis_builder.py --tfs 3 --genes 2

Author: SingleCell-GRN Team
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# CONSTANTS & STYLING
# =============================================================================

class Colors:
    """Modern color palette"""
    # Background
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_LIGHT = "#0f3460"
    
    # Nodes
    TF_FILL = "#4361ee"  # Blue for TFs
    TF_STROKE = "#3a0ca3"
    TF_SELECTED = "#7209b7"
    
    GENE_FILL = "#f72585"  # Pink for genes
    GENE_STROKE = "#b5179e"
    GENE_SELECTED = "#ff006e"
    
    OUTPUT_FILL = "#ffd60a"  # Gold for output genes
    OUTPUT_STROKE = "#ffc300"
    
    # Connections
    ACTIVATOR = "#06d6a0"  # Green
    REPRESSOR = "#ef476f"  # Red
    BIDIRECTIONAL = "#8338ec"  # Purple
    
    # Text
    TEXT_LIGHT = "#ffffff"
    TEXT_DARK = "#1a1a2e"
    TEXT_MUTED = "#a0a0a0"
    
    # Accents
    ACCENT = "#00f5d4"
    GRID = "#2a2a4e"


class ConnectionType(Enum):
    ACTIVATOR = "activator"
    REPRESSOR = "repressor"


@dataclass
class Node:
    """Represents a TF or Gene node"""
    id: str
    x: float
    y: float
    name: str
    is_tf: bool
    is_output: bool = False
    canvas_id: int = -1
    text_id: int = -1


@dataclass
class Connection:
    """Represents a connection between nodes"""
    source_id: str
    target_id: str
    connection_type: ConnectionType
    canvas_ids: List[int] = field(default_factory=list)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class HypothesisBuilder(tk.Tk):
    """Main application window"""
    
    def __init__(self, n_tfs: int = 3, n_genes: int = 2):
        super().__init__()
        
        self.title("🧬 Hypothesis Pathway Builder - SingleCell-GRN")
        self.geometry("1400x900")
        self.configure(bg=Colors.BG_DARK)
        
        # State
        self.nodes: Dict[str, Node] = {}
        self.connections: List[Connection] = []
        self.selected_node: Optional[str] = None
        self.connection_start: Optional[str] = None
        self.drag_data = {"x": 0, "y": 0, "item": None}
        self.node_counter = {"tf": 0, "gene": 0}
        
        # Build UI
        self._create_styles()
        self._create_menu()
        self._create_toolbar()
        self._create_canvas()
        self._create_sidebar()
        self._create_status_bar()
        
        # Initialize with default nodes
        self._initialize_nodes(n_tfs, n_genes)
        
        # Bindings
        self._bind_events()
        
    def _create_styles(self):
        """Create custom ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure(".", background=Colors.BG_DARK, foreground=Colors.TEXT_LIGHT)
        style.configure("TButton", 
                       background=Colors.BG_LIGHT, 
                       foreground=Colors.TEXT_LIGHT,
                       padding=(15, 8),
                       font=('Segoe UI', 10))
        style.map("TButton",
                 background=[('active', Colors.ACCENT)],
                 foreground=[('active', Colors.TEXT_DARK)])
        
        style.configure("Accent.TButton",
                       background=Colors.ACCENT,
                       foreground=Colors.TEXT_DARK)
        
        style.configure("TLabel",
                       background=Colors.BG_DARK,
                       foreground=Colors.TEXT_LIGHT,
                       font=('Segoe UI', 10))
        
        style.configure("Title.TLabel",
                       font=('Segoe UI', 14, 'bold'))
        
        style.configure("TFrame", background=Colors.BG_DARK)
        
    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self, bg=Colors.BG_MEDIUM, fg=Colors.TEXT_LIGHT,
                         activebackground=Colors.ACCENT, activeforeground=Colors.TEXT_DARK)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=Colors.BG_MEDIUM, fg=Colors.TEXT_LIGHT)
        file_menu.add_command(label="New", command=self._new_hypothesis, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self._load_hypothesis, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self._save_hypothesis, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export for Testing...", command=self._export_hypothesis)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0, bg=Colors.BG_MEDIUM, fg=Colors.TEXT_LIGHT)
        edit_menu.add_command(label="Add TF", command=lambda: self._add_node(is_tf=True))
        edit_menu.add_command(label="Add Gene", command=lambda: self._add_node(is_tf=False))
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear All", command=self._clear_all)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=Colors.BG_MEDIUM, fg=Colors.TEXT_LIGHT)
        help_menu.add_command(label="How to Use", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)
        
    def _create_toolbar(self):
        """Create toolbar"""
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=10, pady=5)
        
        # Node buttons
        ttk.Button(toolbar, text="➕ Add TF", 
                  command=lambda: self._add_node(is_tf=True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="➕ Add Gene", 
                  command=lambda: self._add_node(is_tf=False)).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Connection mode buttons
        self.connection_mode = tk.StringVar(value="none")
        
        ttk.Label(toolbar, text="Connection:").pack(side=tk.LEFT, padx=5)
        
        self.btn_activator = ttk.Button(toolbar, text="→ Activator (A)",
                                        command=lambda: self._set_connection_mode("activator"))
        self.btn_activator.pack(side=tk.LEFT, padx=2)
        
        self.btn_repressor = ttk.Button(toolbar, text="⊣ Repressor (R)",
                                        command=lambda: self._set_connection_mode("repressor"))
        self.btn_repressor.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="✋ Select (Esc)",
                  command=lambda: self._set_connection_mode("none")).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Action buttons
        ttk.Button(toolbar, text="🎯 Set Output Gene",
                  command=self._set_output_gene).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Export button
        ttk.Button(toolbar, text="🧪 Test Hypothesis", style="Accent.TButton",
                  command=self._export_hypothesis).pack(side=tk.RIGHT, padx=2)
        
    def _create_canvas(self):
        """Create main canvas for drawing"""
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=Colors.BG_MEDIUM,
            highlightthickness=2,
            highlightbackground=Colors.BG_LIGHT
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw grid
        self._draw_grid()
        
    def _draw_grid(self):
        """Draw background grid"""
        self.canvas.delete("grid")
        w = self.canvas.winfo_width() or 1200
        h = self.canvas.winfo_height() or 700
        
        grid_size = 30
        for x in range(0, w, grid_size):
            self.canvas.create_line(x, 0, x, h, fill=Colors.GRID, tags="grid")
        for y in range(0, h, grid_size):
            self.canvas.create_line(0, y, w, y, fill=Colors.GRID, tags="grid")
            
        self.canvas.tag_lower("grid")
        
    def _create_sidebar(self):
        """Create right sidebar with info and legend"""
        sidebar = ttk.Frame(self, width=280)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)
        sidebar.pack_propagate(False)
        
        # Title
        ttk.Label(sidebar, text="📊 Pathway Info", style="Title.TLabel").pack(pady=10)
        
        # Legend
        legend_frame = ttk.LabelFrame(sidebar, text="Legend")
        legend_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # TF
        tf_legend = tk.Canvas(legend_frame, width=250, height=30, bg=Colors.BG_DARK, highlightthickness=0)
        tf_legend.pack(pady=2)
        tf_legend.create_rectangle(10, 5, 40, 25, fill=Colors.TF_FILL, outline=Colors.TF_STROKE, width=2)
        tf_legend.create_text(130, 15, text="TF (Transcription Factor)", fill=Colors.TEXT_LIGHT, anchor=tk.W)
        
        # Gene
        gene_legend = tk.Canvas(legend_frame, width=250, height=30, bg=Colors.BG_DARK, highlightthickness=0)
        gene_legend.pack(pady=2)
        gene_legend.create_oval(10, 5, 40, 25, fill=Colors.GENE_FILL, outline=Colors.GENE_STROKE, width=2)
        gene_legend.create_text(130, 15, text="Gene", fill=Colors.TEXT_LIGHT, anchor=tk.W)
        
        # Output
        output_legend = tk.Canvas(legend_frame, width=250, height=30, bg=Colors.BG_DARK, highlightthickness=0)
        output_legend.pack(pady=2)
        output_legend.create_oval(10, 5, 40, 25, fill=Colors.OUTPUT_FILL, outline=Colors.OUTPUT_STROKE, width=2)
        output_legend.create_text(130, 15, text="Output Gene (target)", fill=Colors.TEXT_LIGHT, anchor=tk.W)
        
        # Activator
        act_legend = tk.Canvas(legend_frame, width=250, height=30, bg=Colors.BG_DARK, highlightthickness=0)
        act_legend.pack(pady=2)
        act_legend.create_line(10, 15, 40, 15, fill=Colors.ACTIVATOR, width=3, arrow=tk.LAST)
        act_legend.create_text(130, 15, text="Activates (→)", fill=Colors.ACTIVATOR, anchor=tk.W)
        
        # Repressor
        rep_legend = tk.Canvas(legend_frame, width=250, height=30, bg=Colors.BG_DARK, highlightthickness=0)
        rep_legend.pack(pady=2)
        rep_legend.create_line(10, 15, 40, 15, fill=Colors.REPRESSOR, width=3)
        rep_legend.create_line(38, 10, 38, 20, fill=Colors.REPRESSOR, width=3)
        rep_legend.create_text(130, 15, text="Represses (⊣)", fill=Colors.REPRESSOR, anchor=tk.W)
        
        # Stats
        stats_frame = ttk.LabelFrame(sidebar, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.stats_label = ttk.Label(stats_frame, text="TFs: 0\nGenes: 0\nConnections: 0")
        self.stats_label.pack(padx=10, pady=10)
        
        # Instructions
        inst_frame = ttk.LabelFrame(sidebar, text="Quick Tips")
        inst_frame.pack(fill=tk.X, padx=5, pady=5)
        
        instructions = """
• Double-click node to rename
• Drag nodes to move
• Click two nodes to connect
  (select mode first: A or R)
• Right-click to delete
• Set output gene(s) to define
  what you're predicting
        """
        ttk.Label(inst_frame, text=instructions, justify=tk.LEFT).pack(padx=5, pady=5)
        
        # Hypothesis summary
        summary_frame = ttk.LabelFrame(sidebar, text="Hypothesis Summary")
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=10, bg=Colors.BG_LIGHT, 
                                     fg=Colors.TEXT_LIGHT, font=('Consolas', 9),
                                     wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def _create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar(value="Ready. Press 'A' for activator mode, 'R' for repressor mode.")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=10, pady=5)
        
    def _bind_events(self):
        """Bind keyboard and mouse events"""
        # Keyboard
        self.bind('<a>', lambda e: self._set_connection_mode("activator"))
        self.bind('<r>', lambda e: self._set_connection_mode("repressor"))
        self.bind('<Escape>', lambda e: self._set_connection_mode("none"))
        self.bind('<Delete>', self._delete_selected)
        self.bind('<Control-s>', lambda e: self._save_hypothesis())
        self.bind('<Control-o>', lambda e: self._load_hypothesis())
        self.bind('<Control-n>', lambda e: self._new_hypothesis())
        
        # Canvas events
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        self.canvas.bind('<Double-Button-1>', self._on_double_click)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Configure>', lambda e: self._draw_grid())
        
    def _initialize_nodes(self, n_tfs: int, n_genes: int):
        """Create initial TF and Gene nodes"""
        canvas_width = 1000
        canvas_height = 600
        
        # TFs on the left
        tf_spacing = canvas_height // (n_tfs + 1)
        for i in range(n_tfs):
            self._add_node(
                is_tf=True,
                x=150,
                y=(i + 1) * tf_spacing,
                name=f"TF_{i+1}"
            )
        
        # Genes on the right
        gene_spacing = canvas_height // (n_genes + 1)
        for i in range(n_genes):
            self._add_node(
                is_tf=False,
                x=canvas_width - 150,
                y=(i + 1) * gene_spacing,
                name=f"Gene_{i+1}"
            )
            
    def _add_node(self, is_tf: bool, x: float = None, y: float = None, name: str = None):
        """Add a new node to the canvas"""
        if x is None:
            x = 400 + (50 if is_tf else -50)
        if y is None:
            y = 300
            
        node_type = "tf" if is_tf else "gene"
        self.node_counter[node_type] += 1
        
        if name is None:
            name = f"{'TF' if is_tf else 'Gene'}_{self.node_counter[node_type]}"
        
        node_id = f"{node_type}_{self.node_counter[node_type]}"
        
        node = Node(
            id=node_id,
            x=x,
            y=y,
            name=name,
            is_tf=is_tf
        )
        
        self._draw_node(node)
        self.nodes[node_id] = node
        self._update_stats()
        
    def _draw_node(self, node: Node):
        """Draw a node on the canvas"""
        size = 40 if node.is_tf else 35
        
        if node.is_tf:
            # Square for TF
            fill = Colors.TF_FILL
            outline = Colors.TF_STROKE
            if node.is_output:
                fill = Colors.OUTPUT_FILL
                outline = Colors.OUTPUT_STROKE
            
            node.canvas_id = self.canvas.create_rectangle(
                node.x - size, node.y - size,
                node.x + size, node.y + size,
                fill=fill, outline=outline, width=3,
                tags=("node", node.id)
            )
        else:
            # Circle for Gene
            fill = Colors.GENE_FILL
            outline = Colors.GENE_STROKE
            if node.is_output:
                fill = Colors.OUTPUT_FILL
                outline = Colors.OUTPUT_STROKE
            
            node.canvas_id = self.canvas.create_oval(
                node.x - size, node.y - size,
                node.x + size, node.y + size,
                fill=fill, outline=outline, width=3,
                tags=("node", node.id)
            )
        
        # Node label
        node.text_id = self.canvas.create_text(
            node.x, node.y,
            text=node.name,
            fill=Colors.TEXT_LIGHT,
            font=('Segoe UI', 10, 'bold'),
            tags=("node_text", node.id)
        )
        
    def _set_connection_mode(self, mode: str):
        """Set the connection mode"""
        self.connection_mode.set(mode)
        self.connection_start = None
        
        if mode == "activator":
            self.status_var.set("🟢 ACTIVATOR MODE: Click source node, then target node to create activation")
            self.config(cursor="crosshair")
        elif mode == "repressor":
            self.status_var.set("🔴 REPRESSOR MODE: Click source node, then target node to create repression")
            self.config(cursor="crosshair")
        else:
            self.status_var.set("Ready. Drag nodes to move, double-click to rename.")
            self.config(cursor="arrow")
            
    def _on_canvas_click(self, event):
        """Handle canvas click"""
        # Find clicked node
        clicked = self.canvas.find_closest(event.x, event.y)
        tags = self.canvas.gettags(clicked)
        
        node_id = None
        for tag in tags:
            if tag.startswith("tf_") or tag.startswith("gene_"):
                node_id = tag
                break
        
        if self.connection_mode.get() != "none" and node_id:
            # Connection mode
            if self.connection_start is None:
                # First node
                self.connection_start = node_id
                self._highlight_node(node_id)
                self.status_var.set(f"Selected {self.nodes[node_id].name}. Now click target node.")
            else:
                # Second node - create connection
                if node_id != self.connection_start:
                    conn_type = ConnectionType.ACTIVATOR if self.connection_mode.get() == "activator" else ConnectionType.REPRESSOR
                    self._add_connection(self.connection_start, node_id, conn_type)
                
                self._unhighlight_node(self.connection_start)
                self.connection_start = None
                self._set_connection_mode("none")
        elif node_id:
            # Selection mode - prepare for drag
            self.drag_data["item"] = node_id
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            self.selected_node = node_id
            
    def _on_double_click(self, event):
        """Handle double-click to rename node"""
        clicked = self.canvas.find_closest(event.x, event.y)
        tags = self.canvas.gettags(clicked)
        
        node_id = None
        for tag in tags:
            if tag.startswith("tf_") or tag.startswith("gene_"):
                node_id = tag
                break
        
        if node_id:
            node = self.nodes[node_id]
            new_name = simpledialog.askstring(
                "Rename Node",
                f"Enter new name for {node.name}:",
                initialvalue=node.name,
                parent=self
            )
            if new_name:
                node.name = new_name
                self.canvas.itemconfig(node.text_id, text=new_name)
                self._update_summary()
                
    def _on_drag(self, event):
        """Handle node dragging"""
        if self.drag_data["item"]:
            node_id = self.drag_data["item"]
            node = self.nodes[node_id]
            
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            
            # Move node
            self.canvas.move(node.canvas_id, dx, dy)
            self.canvas.move(node.text_id, dx, dy)
            
            node.x += dx
            node.y += dy
            
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            
            # Update connections
            self._redraw_connections()
            
    def _on_release(self, event):
        """Handle mouse release"""
        self.drag_data["item"] = None
        
    def _on_right_click(self, event):
        """Handle right-click to delete"""
        clicked = self.canvas.find_closest(event.x, event.y)
        tags = self.canvas.gettags(clicked)
        
        # Check if clicked on a node
        node_id = None
        for tag in tags:
            if tag.startswith("tf_") or tag.startswith("gene_"):
                node_id = tag
                break
        
        if node_id:
            # Show context menu
            menu = tk.Menu(self, tearoff=0, bg=Colors.BG_MEDIUM, fg=Colors.TEXT_LIGHT)
            menu.add_command(label=f"Delete {self.nodes[node_id].name}", 
                           command=lambda: self._delete_node(node_id))
            menu.add_command(label="Set as Output", 
                           command=lambda: self._toggle_output(node_id))
            menu.tk_popup(event.x_root, event.y_root)
        else:
            # Check if clicked on a connection
            for conn in self.connections:
                for cid in conn.canvas_ids:
                    if cid in clicked:
                        menu = tk.Menu(self, tearoff=0, bg=Colors.BG_MEDIUM, fg=Colors.TEXT_LIGHT)
                        menu.add_command(label="Delete Connection", 
                                       command=lambda c=conn: self._delete_connection(c))
                        menu.tk_popup(event.x_root, event.y_root)
                        return
                        
    def _add_connection(self, source_id: str, target_id: str, conn_type: ConnectionType):
        """Add a connection between nodes"""
        # Check if connection already exists
        for conn in self.connections:
            if conn.source_id == source_id and conn.target_id == target_id:
                messagebox.showwarning("Connection Exists", "This connection already exists!")
                return
        
        conn = Connection(source_id, target_id, conn_type)
        self.connections.append(conn)
        self._draw_connection(conn)
        self._update_stats()
        self._update_summary()
        
    def _draw_connection(self, conn: Connection):
        """Draw a connection on the canvas"""
        source = self.nodes[conn.source_id]
        target = self.nodes[conn.target_id]
        
        # Calculate arrow points
        dx = target.x - source.x
        dy = target.y - source.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist == 0:
            return
            
        # Offset from center
        offset = 45
        sx = source.x + (dx / dist) * offset
        sy = source.y + (dy / dist) * offset
        tx = target.x - (dx / dist) * offset
        ty = target.y - (dy / dist) * offset
        
        color = Colors.ACTIVATOR if conn.connection_type == ConnectionType.ACTIVATOR else Colors.REPRESSOR
        
        # Draw line
        line_id = self.canvas.create_line(
            sx, sy, tx, ty,
            fill=color, width=3, arrow=tk.LAST, arrowshape=(12, 15, 5),
            tags=("connection",)
        )
        conn.canvas_ids.append(line_id)
        
        # For repressor, add bar at the end
        if conn.connection_type == ConnectionType.REPRESSOR:
            # Calculate perpendicular
            px = -dy / dist * 8
            py = dx / dist * 8
            bar_id = self.canvas.create_line(
                tx - px, ty - py, tx + px, ty + py,
                fill=color, width=4, tags=("connection",)
            )
            conn.canvas_ids.append(bar_id)
        
        # Lower connections below nodes
        self.canvas.tag_lower("connection")
        self.canvas.tag_lower("grid")
        
    def _redraw_connections(self):
        """Redraw all connections"""
        for conn in self.connections:
            for cid in conn.canvas_ids:
                self.canvas.delete(cid)
            conn.canvas_ids = []
            self._draw_connection(conn)
            
    def _highlight_node(self, node_id: str):
        """Highlight a selected node"""
        node = self.nodes[node_id]
        color = Colors.TF_SELECTED if node.is_tf else Colors.GENE_SELECTED
        self.canvas.itemconfig(node.canvas_id, outline=color, width=5)
        
    def _unhighlight_node(self, node_id: str):
        """Remove highlight from node"""
        node = self.nodes[node_id]
        color = Colors.TF_STROKE if node.is_tf else Colors.GENE_STROKE
        if node.is_output:
            color = Colors.OUTPUT_STROKE
        self.canvas.itemconfig(node.canvas_id, outline=color, width=3)
        
    def _toggle_output(self, node_id: str):
        """Toggle output gene status"""
        node = self.nodes[node_id]
        node.is_output = not node.is_output
        
        # Redraw node with new color
        self.canvas.delete(node.canvas_id)
        self.canvas.delete(node.text_id)
        self._draw_node(node)
        
        self._update_summary()
        
    def _set_output_gene(self):
        """Set output gene via dialog"""
        # Get all genes
        genes = [n for n in self.nodes.values() if not n.is_tf]
        if not genes:
            messagebox.showwarning("No Genes", "Add some genes first!")
            return
            
        # Create selection dialog
        dialog = tk.Toplevel(self)
        dialog.title("Select Output Gene(s)")
        dialog.geometry("300x400")
        dialog.configure(bg=Colors.BG_DARK)
        
        ttk.Label(dialog, text="Select output gene(s):").pack(pady=10)
        
        listbox = tk.Listbox(dialog, selectmode=tk.MULTIPLE, bg=Colors.BG_LIGHT,
                            fg=Colors.TEXT_LIGHT, selectbackground=Colors.ACCENT)
        for gene in genes:
            listbox.insert(tk.END, gene.name)
            if gene.is_output:
                listbox.selection_set(genes.index(gene))
        listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        def apply():
            # Reset all outputs
            for gene in genes:
                gene.is_output = False
            # Set selected as outputs
            for idx in listbox.curselection():
                genes[idx].is_output = True
            # Redraw
            for gene in genes:
                self.canvas.delete(gene.canvas_id)
                self.canvas.delete(gene.text_id)
                self._draw_node(gene)
            self._update_summary()
            dialog.destroy()
        
        ttk.Button(dialog, text="Apply", command=apply).pack(pady=10)
        
    def _delete_node(self, node_id: str):
        """Delete a node and its connections"""
        node = self.nodes[node_id]
        
        # Delete connections
        self.connections = [c for c in self.connections 
                          if c.source_id != node_id and c.target_id != node_id]
        
        # Delete from canvas
        self.canvas.delete(node.canvas_id)
        self.canvas.delete(node.text_id)
        
        del self.nodes[node_id]
        self._redraw_connections()
        self._update_stats()
        self._update_summary()
        
    def _delete_connection(self, conn: Connection):
        """Delete a connection"""
        for cid in conn.canvas_ids:
            self.canvas.delete(cid)
        self.connections.remove(conn)
        self._update_stats()
        self._update_summary()
        
    def _delete_selected(self, event=None):
        """Delete selected node"""
        if self.selected_node:
            self._delete_node(self.selected_node)
            self.selected_node = None
            
    def _update_stats(self):
        """Update statistics display"""
        n_tfs = sum(1 for n in self.nodes.values() if n.is_tf)
        n_genes = sum(1 for n in self.nodes.values() if not n.is_tf)
        n_outputs = sum(1 for n in self.nodes.values() if n.is_output)
        
        self.stats_label.config(text=f"TFs: {n_tfs}\nGenes: {n_genes}\nConnections: {len(self.connections)}\nOutput Genes: {n_outputs}")
        
    def _update_summary(self):
        """Update hypothesis summary text"""
        self.summary_text.delete("1.0", tk.END)
        
        if not self.connections:
            self.summary_text.insert(tk.END, "No connections yet.\n\nAdd connections to define your hypothesis.")
            return
        
        # Build summary
        lines = ["HYPOTHESIS:\n"]
        for conn in self.connections:
            source = self.nodes[conn.source_id].name
            target = self.nodes[conn.target_id].name
            symbol = "→" if conn.connection_type == ConnectionType.ACTIVATOR else "⊣"
            action = "activates" if conn.connection_type == ConnectionType.ACTIVATOR else "represses"
            lines.append(f"{source} {symbol} {target}\n  ({source} {action} {target})\n")
        
        # Output genes
        outputs = [n.name for n in self.nodes.values() if n.is_output]
        if outputs:
            lines.append(f"\nOUTPUT GENES:\n  {', '.join(outputs)}")
        
        self.summary_text.insert(tk.END, "".join(lines))
        
    def _clear_all(self):
        """Clear all nodes and connections"""
        if messagebox.askyesno("Clear All", "Are you sure you want to clear everything?"):
            self.canvas.delete("node")
            self.canvas.delete("node_text")
            self.canvas.delete("connection")
            self.nodes.clear()
            self.connections.clear()
            self.node_counter = {"tf": 0, "gene": 0}
            self._update_stats()
            self._update_summary()
            
    def _new_hypothesis(self):
        """Create new hypothesis"""
        self._clear_all()
        
    def _save_hypothesis(self):
        """Save hypothesis to JSON file"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Hypothesis"
        )
        
        if filepath:
            data = {
                "nodes": [
                    {"id": n.id, "name": n.name, "x": n.x, "y": n.y, 
                     "is_tf": n.is_tf, "is_output": n.is_output}
                    for n in self.nodes.values()
                ],
                "connections": [
                    {"source": c.source_id, "target": c.target_id, 
                     "type": c.connection_type.value}
                    for c in self.connections
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.status_var.set(f"Saved to {filepath}")
            
    def _load_hypothesis(self):
        """Load hypothesis from JSON file"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Hypothesis"
        )
        
        if filepath:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self._clear_all()
            
            # Load nodes
            for node_data in data["nodes"]:
                is_tf = node_data["is_tf"]
                node_type = "tf" if is_tf else "gene"
                self.node_counter[node_type] += 1
                
                node = Node(
                    id=node_data["id"],
                    x=node_data["x"],
                    y=node_data["y"],
                    name=node_data["name"],
                    is_tf=is_tf,
                    is_output=node_data.get("is_output", False)
                )
                self._draw_node(node)
                self.nodes[node.id] = node
            
            # Load connections
            for conn_data in data["connections"]:
                conn_type = ConnectionType.ACTIVATOR if conn_data["type"] == "activator" else ConnectionType.REPRESSOR
                conn = Connection(conn_data["source"], conn_data["target"], conn_type)
                self.connections.append(conn)
                self._draw_connection(conn)
            
            self._update_stats()
            self._update_summary()
            self.status_var.set(f"Loaded from {filepath}")
            
    def _export_hypothesis(self):
        """Export hypothesis for testing with GRN model"""
        if not self.connections:
            messagebox.showwarning("No Hypothesis", "Create some connections first!")
            return
        
        outputs = [n.name for n in self.nodes.values() if n.is_output]
        if not outputs:
            messagebox.showwarning("No Output", "Set at least one output gene!")
            return
        
        # Generate hypothesis text file
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Hypothesis for Testing"
        )
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write("# Hypothesis generated by SingleCell-GRN Pathway Builder\n")
                f.write("# Format: SOURCE -> TARGET (activator) or SOURCE -| TARGET (repressor)\n\n")
                
                f.write("# NODES\n")
                for node in self.nodes.values():
                    node_type = "TF" if node.is_tf else "GENE"
                    output_tag = " [OUTPUT]" if node.is_output else ""
                    f.write(f"# {node_type}: {node.name}{output_tag}\n")
                
                f.write("\n# CONNECTIONS\n")
                for conn in self.connections:
                    source = self.nodes[conn.source_id].name
                    target = self.nodes[conn.target_id].name
                    symbol = "->" if conn.connection_type == ConnectionType.ACTIVATOR else "-|"
                    f.write(f"{source} {symbol} {target}\n")
                
                f.write("\n# OUTPUT GENES\n")
                for output in outputs:
                    f.write(f"OUTPUT: {output}\n")
            
            # Show instructions
            msg = f"""Hypothesis exported to:
{filepath}

To test this hypothesis, run:

python scripts/finetune_gene.py \\
    --gene {outputs[0]} \\
    --hypothesis {filepath} \\
    --model results/sclc/ \\
    --singlecell data/sclc/preprocessed.h5ad \\
    --tf-binding data/tf_binding/ \\
    --output results/hypothesis_test/
"""
            messagebox.showinfo("Export Complete", msg)
            self.status_var.set(f"Exported to {filepath}")
            
    def _show_help(self):
        """Show help dialog"""
        help_text = """
🧬 HYPOTHESIS PATHWAY BUILDER - Help

ADDING NODES:
• Click "Add TF" or "Add Gene" to add nodes
• TFs appear as squares (■)
• Genes appear as circles (●)

RENAMING NODES:
• Double-click any node to rename it
• Use real gene symbols (e.g., ASCL1, SSTR2)

CREATING CONNECTIONS:
• Press 'A' for activator mode (→)
• Press 'R' for repressor mode (⊣)
• Click source node, then target node
• Press 'Esc' to exit connection mode

EDITING:
• Drag nodes to reposition
• Right-click to delete nodes/connections
• Select output genes via toolbar

EXPORTING:
• Click "Test Hypothesis" to export
• Use the generated file with finetune_gene.py

KEYBOARD SHORTCUTS:
• A = Activator mode
• R = Repressor mode
• Esc = Select mode
• Delete = Delete selected
• Ctrl+S = Save
• Ctrl+O = Open
• Ctrl+N = New
        """
        messagebox.showinfo("Help", help_text)
        
    def _show_about(self):
        """Show about dialog"""
        about_text = """
🧬 HYPOTHESIS PATHWAY BUILDER

Part of SingleCell-GRN Framework

Build visual regulatory hypotheses and test them
against your trained GRN model.

Version: 1.0
        """
        messagebox.showinfo("About", about_text)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hypothesis Pathway Builder GUI")
    parser.add_argument('--tfs', type=int, default=3, help='Initial number of TFs')
    parser.add_argument('--genes', type=int, default=2, help='Initial number of genes')
    
    args = parser.parse_args()
    
    app = HypothesisBuilder(n_tfs=args.tfs, n_genes=args.genes)
    app.mainloop()


if __name__ == '__main__':
    main()

