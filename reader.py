#!/usr/bin/env python3
"""
G-S 3D Reconstruction Utility - Complete PNG Support with Scrolling
Author: Angledcrystals
Date: 2025-06-08
Version: 1.3

A specialized utility for processing G-S Depth Mapper PNG output into 3D reconstructions,
point clouds, meshes, and interactive visualizations.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import pickle
import time
from datetime import datetime
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata
from PIL import Image

# Fixed sklearn import
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    try:
        import sklearn
        from sklearn.cluster import DBSCAN
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False
        print("scikit-learn not available - clustering features will be disabled")

import cv2

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not available - some 3D features will be disabled")

class GS3DReconstructorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("G-S 3D Reconstructor - PNG Support v1.3")
        self.root.geometry("1600x1000")
        
        # Data storage
        self.alignment_data = []
        self.depth_map_data = None
        self.depth_map_original = None  # Store original for reference
        self.point_cloud_3d = None
        self.mesh_vertices = None
        self.mesh_faces = None
        self.quality_weights = None
        self.condition_colors = None
        
        # PNG processing parameters
        self.depth_scale_factor = tk.DoubleVar(value=1.0)
        self.depth_invert = tk.BooleanVar(value=False)
        self.depth_normalize = tk.BooleanVar(value=True)
        self.png_colormap_mode = tk.StringVar(value="grayscale")
        
        # 3D Reconstruction parameters
        self.reconstruction_method = tk.StringVar(value="point_cloud")
        self.mesh_resolution = tk.IntVar(value=128)
        self.quality_threshold = tk.DoubleVar(value=0.1)
        self.smoothing_iterations = tk.IntVar(value=3)
        self.clustering_enabled = tk.BooleanVar(value=False)
        self.cluster_eps = tk.DoubleVar(value=0.1)
        self.noise_removal = tk.BooleanVar(value=True)
        
        # Visualization parameters
        self.colormap_3d = tk.StringVar(value="viridis")
        self.point_size = tk.DoubleVar(value=2.0)
        self.transparency = tk.DoubleVar(value=0.8)
        self.lighting_enabled = tk.BooleanVar(value=True)
        
        # Export parameters
        self.export_format = tk.StringVar(value="PLY")
        self.include_colors = tk.BooleanVar(value=True)
        self.include_normals = tk.BooleanVar(value=True)
        self.compress_output = tk.BooleanVar(value=False)
        
        self.setup_gui()
        
    def setup_scrollable_controls(self, parent):
        """Add scrollbar to existing control panel."""
        # Create canvas and scrollbar with fixed width
        canvas = tk.Canvas(parent, highlightthickness=0, width=350)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        # Bind mouse wheel events
        scrollable_frame.bind('<Enter>', _bind_mousewheel)
        scrollable_frame.bind('<Leave>', _unbind_mousewheel)
        canvas.bind('<Enter>', _bind_mousewheel)
        canvas.bind('<Leave>', _unbind_mousewheel)
        
        return scrollable_frame
        
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Create main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollable_control_frame = self.setup_scrollable_controls(control_frame)
        self.setup_control_panel(scrollable_control_frame)
        self.setup_3d_visualization(viz_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel with all reconstruction options."""
        
        # Title with PNG support
        title_text = "G-S 3D Reconstructor (PNG Support)"
        title_label = ttk.Label(parent, text=title_text, font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Dependency status
        status_text = f"Dependencies: "
        status_text += f"✅ NumPy " if 'numpy' in globals() else "❌ NumPy "
        status_text += f"✅ PIL " if 'PIL' in globals() else "❌ PIL "
        status_text += f"✅ scikit-learn " if SKLEARN_AVAILABLE else "❌ scikit-learn "
        status_text += f"✅ Open3D " if OPEN3D_AVAILABLE else "❌ Open3D "
        
        status_label = ttk.Label(parent, text=status_text, font=("Arial", 8))
        status_label.pack(pady=(0, 20))
        
        # Data Loading Section
        data_frame = ttk.LabelFrame(parent, text="Step 1: Load G-S Data", padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(data_frame, text="📁 Load Alignment Data (.txt)", 
                  command=self.load_alignment_data, width=30).pack(fill=tk.X, pady=2)
        
        ttk.Button(data_frame, text="🖼️ Load Depth Map (.png)", 
                  command=self.load_depth_map_png, width=30).pack(fill=tk.X, pady=2)
        
        ttk.Button(data_frame, text="🔄 Load Both (Auto-detect)", 
                  command=self.load_combined_data, width=30).pack(fill=tk.X, pady=2)
        
        self.data_status_label = ttk.Label(data_frame, text="No data loaded", 
                                          foreground="red")
        self.data_status_label.pack(pady=5)
        
        # PNG Processing Section
        png_frame = ttk.LabelFrame(parent, text="Step 2: PNG Depth Processing", padding=10)
        png_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(png_frame, text="PNG Colormap Mode:").pack(anchor=tk.W)
        colormap_mode_combo = ttk.Combobox(png_frame, textvariable=self.png_colormap_mode,
                                          values=["grayscale", "viridis", "plasma", "hot", "coolwarm"])
        colormap_mode_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(png_frame, text="Depth Scale Factor:").pack(anchor=tk.W)
        depth_scale_scale = ttk.Scale(png_frame, from_=0.1, to=10.0, 
                                     variable=self.depth_scale_factor, orient=tk.HORIZONTAL)
        depth_scale_scale.pack(fill=tk.X, pady=2)
        ttk.Label(png_frame, textvariable=self.depth_scale_factor).pack(anchor=tk.W)
        
        ttk.Checkbutton(png_frame, text="Invert depth values", 
                       variable=self.depth_invert).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(png_frame, text="Normalize depth to [0,1]", 
                       variable=self.depth_normalize).pack(anchor=tk.W, pady=2)
        
        self.reprocess_png_button = ttk.Button(png_frame, text="🔄 Reprocess PNG", 
                                              command=self.reprocess_png_depth, 
                                              state="disabled")
        self.reprocess_png_button.pack(fill=tk.X, pady=5)
        
        # Reconstruction Method Section
        method_frame = ttk.LabelFrame(parent, text="Step 3: Reconstruction Method", padding=10)
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        methods = [
            ("Point Cloud (Fast)", "point_cloud"),
            ("Surface Mesh (Delaunay)", "delaunay_mesh"),
            ("PNG Height Field", "png_heightfield"),
            ("Hybrid (Points + Mesh)", "hybrid"),
            ("Multi-Scale Reconstruction", "multi_scale")
        ]
        
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.reconstruction_method, 
                           value=value).pack(anchor=tk.W)
        
        # Quality and Filtering Section
        quality_frame = ttk.LabelFrame(parent, text="Step 4: Quality Control", padding=10)
        quality_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(quality_frame, text="Quality Threshold:").pack(anchor=tk.W)
        quality_scale = ttk.Scale(quality_frame, from_=0.01, to=1.0, 
                                 variable=self.quality_threshold, orient=tk.HORIZONTAL)
        quality_scale.pack(fill=tk.X, pady=2)
        ttk.Label(quality_frame, textvariable=self.quality_threshold).pack(anchor=tk.W)
        
        if SKLEARN_AVAILABLE:
            ttk.Checkbutton(quality_frame, text="Enable clustering (DBSCAN)", 
                           variable=self.clustering_enabled).pack(anchor=tk.W, pady=2)
            
            ttk.Label(quality_frame, text="Cluster Epsilon:").pack(anchor=tk.W)
            cluster_scale = ttk.Scale(quality_frame, from_=0.01, to=0.5, 
                                     variable=self.cluster_eps, orient=tk.HORIZONTAL)
            cluster_scale.pack(fill=tk.X, pady=2)
        else:
            ttk.Label(quality_frame, text="Clustering disabled (scikit-learn not available)", 
                     foreground="orange").pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(quality_frame, text="Remove noise points", 
                       variable=self.noise_removal).pack(anchor=tk.W, pady=2)
        
        # Mesh Parameters Section
        mesh_frame = ttk.LabelFrame(parent, text="Step 5: Mesh Parameters", padding=10)
        mesh_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mesh_frame, text="Mesh Resolution:").pack(anchor=tk.W)
        resolution_scale = ttk.Scale(mesh_frame, from_=32, to=512, 
                                    variable=self.mesh_resolution, orient=tk.HORIZONTAL)
        resolution_scale.pack(fill=tk.X, pady=2)
        ttk.Label(mesh_frame, textvariable=self.mesh_resolution).pack(anchor=tk.W)
        
        ttk.Label(mesh_frame, text="Smoothing Iterations:").pack(anchor=tk.W)
        smooth_scale = ttk.Scale(mesh_frame, from_=0, to=10, 
                                variable=self.smoothing_iterations, orient=tk.HORIZONTAL)
        smooth_scale.pack(fill=tk.X, pady=2)
        ttk.Label(mesh_frame, textvariable=self.smoothing_iterations).pack(anchor=tk.W)
        
        # Visualization Section
        viz_control_frame = ttk.LabelFrame(parent, text="Step 6: Visualization", padding=10)
        viz_control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(viz_control_frame, text="3D Colormap:").pack(anchor=tk.W)
        colormap_combo = ttk.Combobox(viz_control_frame, textvariable=self.colormap_3d,
                                     values=["viridis", "plasma", "rainbow", "coolwarm", 
                                            "depth", "quality", "condition"])
        colormap_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(viz_control_frame, text="Point Size:").pack(anchor=tk.W)
        size_scale = ttk.Scale(viz_control_frame, from_=0.5, to=5.0, 
                              variable=self.point_size, orient=tk.HORIZONTAL)
        size_scale.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(viz_control_frame, text="Enable lighting", 
                       variable=self.lighting_enabled).pack(anchor=tk.W, pady=2)
        
        # Processing Section
        process_frame = ttk.LabelFrame(parent, text="Step 7: Process", padding=10)
        process_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.process_button = ttk.Button(process_frame, text="🔄 Generate 3D Reconstruction", 
                                        command=self.generate_3d_reconstruction, 
                                        state="disabled")
        self.process_button.pack(fill=tk.X, pady=5)
        
        self.analyze_button = ttk.Button(process_frame, text="📊 Analyze Quality Distribution", 
                                        command=self.analyze_quality_distribution, 
                                        state="disabled")
        self.analyze_button.pack(fill=tk.X, pady=2)
        
        self.process_status_label = ttk.Label(process_frame, text="Load data first", 
                                             foreground="orange")
        self.process_status_label.pack(pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(process_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Export Section
        export_frame = ttk.LabelFrame(parent, text="Step 8: Export 3D Data", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(export_frame, text="Export Format:").pack(anchor=tk.W)
        format_combo = ttk.Combobox(export_frame, textvariable=self.export_format,
                                   values=["PLY", "OBJ", "STL", "JSON", "CSV"])
        format_combo.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(export_frame, text="Include colors", 
                       variable=self.include_colors).pack(anchor=tk.W)
        ttk.Checkbutton(export_frame, text="Include normals", 
                       variable=self.include_normals).pack(anchor=tk.W)
        ttk.Checkbutton(export_frame, text="Compress output", 
                       variable=self.compress_output).pack(anchor=tk.W)
        
        self.export_button = ttk.Button(export_frame, text="💾 Export 3D Model", 
                                       command=self.export_3d_model, 
                                       state="disabled")
        self.export_button.pack(fill=tk.X, pady=5)
        
        if OPEN3D_AVAILABLE:
            self.open3d_button = ttk.Button(export_frame, text="🎮 Open in Open3D Viewer", 
                                           command=self.open_in_open3d, 
                                           state="disabled")
            self.open3d_button.pack(fill=tk.X, pady=2)
        
        # Statistics Section
        stats_frame = ttk.LabelFrame(parent, text="Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_label = ttk.Label(stats_frame, text="No reconstruction generated", 
                                    font=("Courier", 8))
        self.stats_label.pack(fill=tk.X)
        
    def setup_3d_visualization(self, parent):
        """Setup the 3D visualization panel."""
        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        self.show_welcome_plot()
        
    def show_welcome_plot(self):
        """Show welcome message in 3D plot."""
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d')
        
        # Create a sample 3D visualization
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        theta, phi = np.meshgrid(theta, phi)
        
        x = np.sin(phi) * np.cos(theta) * 0.5
        y = np.sin(phi) * np.sin(theta) * 0.5
        z = np.cos(phi) * 0.5
        
        ax.plot_surface(x, y, z, alpha=0.3, color='cyan')
        ax.text(0, 0, 0.8, "G-S 3D Reconstructor v1.3\nby Angledcrystals\n\nPNG Depth Map Support\nLoad PNG depth maps from your\nenhanced G-S tool!", 
                ha='center', va='center', fontsize=12, weight='bold')
        
        ax.set_xlabel('X (G_refl_x)')
        ax.set_ylabel('Y (G_refl_y)')
        ax.set_zlabel('Z (G_refl_z)')
        ax.set_title("G-S 3D Reconstruction Utility - PNG Support")
        
        self.canvas.draw()
        
    def load_alignment_data(self):
        """Load G-S alignment data file."""
        filename = filedialog.askopenfilename(
            title="Select G-S Alignment Data File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.parse_alignment_file(filename)
                self.data_status_label.config(
                    text=f"✅ Alignment data: {len(self.alignment_data)} points", 
                    foreground="green"
                )
                self.enable_processing_buttons()
                messagebox.showinfo("Success", f"Loaded {len(self.alignment_data)} alignment points")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load alignment data: {str(e)}")
                
    def load_depth_map_png(self):
        """Load depth map from PNG file."""
        filename = filedialog.askopenfilename(
            title="Select PNG Depth Map File",
            filetypes=[("PNG files", "*.png"), ("Image files", "*.png *.jpg *.jpeg *.tiff *.tif"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.process_png_depth_map(filename)
                self.data_status_label.config(
                    text=f"✅ PNG depth map: {self.depth_map_data.shape}", 
                    foreground="green"
                )
                self.enable_processing_buttons()
                self.reprocess_png_button.config(state="normal")
                messagebox.showinfo("Success", f"Loaded PNG depth map: {self.depth_map_data.shape}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load PNG depth map: {str(e)}")
                
    def process_png_depth_map(self, filename):
        """Process PNG depth map into usable depth data."""
        # Load PNG image
        pil_image = Image.open(filename)
        
        # Convert to numpy array
        if pil_image.mode == 'RGBA':
            # Convert RGBA to RGB
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'P':
            # Convert palette to RGB
            pil_image = pil_image.convert('RGB')
        
        img_array = np.array(pil_image)
        
        # Store original for reference
        self.depth_map_original = img_array.copy()
        
        # Convert to depth values
        if len(img_array.shape) == 3:
            # Color image - convert to grayscale using luminosity
            depth_map = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        else:
            # Already grayscale
            depth_map = img_array.astype(np.float64)
        
        # Convert from 0-255 to 0-1 range
        depth_map = depth_map / 255.0
        
        # Apply processing options
        if self.depth_invert.get():
            depth_map = 1.0 - depth_map
        
        # Apply scaling
        depth_map = depth_map * self.depth_scale_factor.get()
        
        # Normalize if requested
        if self.depth_normalize.get():
            if depth_map.max() > depth_map.min():
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        self.depth_map_data = depth_map
        
        print(f"Processed PNG depth map: {depth_map.shape}, range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
    def reprocess_png_depth(self):
        """Reprocess PNG with current settings."""
        if self.depth_map_original is None:
            messagebox.showwarning("No PNG", "Load a PNG depth map first")
            return
        
        try:
            # Reprocess from original
            img_array = self.depth_map_original
            
            # Convert to depth values
            if len(img_array.shape) == 3:
                depth_map = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            else:
                depth_map = img_array.astype(np.float64)
            
            depth_map = depth_map / 255.0
            
            if self.depth_invert.get():
                depth_map = 1.0 - depth_map
            
            depth_map = depth_map * self.depth_scale_factor.get()
            
            if self.depth_normalize.get():
                if depth_map.max() > depth_map.min():
                    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
            self.depth_map_data = depth_map
            
            self.data_status_label.config(
                text=f"✅ PNG reprocessed: {self.depth_map_data.shape}, range: [{depth_map.min():.3f}, {depth_map.max():.3f}]", 
                foreground="green"
            )
            
            messagebox.showinfo("Success", f"PNG reprocessed with current settings")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reprocess PNG: {str(e)}")
                
    def load_combined_data(self):
        """Load both alignment data and PNG depth map from directory."""
        directory = filedialog.askdirectory(title="Select directory with G-S output files")
        if directory:
            try:
                loaded_files = []
                
                # Look for alignment data files
                for file in os.listdir(directory):
                    if file.endswith('.txt') and ('alignment' in file.lower() or 'gs' in file.lower()):
                        self.parse_alignment_file(os.path.join(directory, file))
                        loaded_files.append(f"Alignment: {file}")
                        break
                
                # Look for PNG depth map files
                for file in os.listdir(directory):
                    if file.endswith('.png') and ('depth' in file.lower() or 'map' in file.lower()):
                        self.process_png_depth_map(os.path.join(directory, file))
                        loaded_files.append(f"PNG depth map: {file}")
                        self.reprocess_png_button.config(state="normal")
                        break
                
                if loaded_files:
                    status_text = " + ".join(loaded_files)
                    self.data_status_label.config(text=f"✅ {status_text}", foreground="green")
                    self.enable_processing_buttons()
                    messagebox.showinfo("Success", f"Loaded files:\n" + "\n".join(loaded_files))
                else:
                    messagebox.showwarning("No Files", "No G-S data files found in directory")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load combined data: {str(e)}")
                
    def parse_alignment_file(self, filename):
        """Parse G-S alignment file and extract 3D coordinates."""
        with open(filename, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        alignments = []
        current_condition = None
        parsing_data = False
        
        for line in lines:
            line = line.strip()
            if "CONDITION:" in line and "PROJECTION" in line:
                current_condition = line.replace("CONDITION: ", "").strip()
                parsing_data = False
                continue
            if line.startswith("G_theta_deg,G_phi_deg,Hadit_theta_deg"):
                parsing_data = True
                continue
            if parsing_data and line and not line.startswith("=") and "," in line:
                try:
                    parts = line.split(',')
                    if len(parts) >= 11:
                        alignment = {
                            'condition': current_condition,
                            'G_theta': float(parts[0]), 'G_phi': float(parts[1]),
                            'Hadit_theta': float(parts[2]), 'Hadit_phi': float(parts[3]),
                            'S_x': float(parts[4]), 'S_y': float(parts[5]),
                            'dist_boundary': float(parts[6]), 'dist_origin': float(parts[7]),
                            'G_refl_x': float(parts[8]), 'G_refl_y': float(parts[9]),
                            'G_refl_z': float(parts[10])
                        }
                        alignments.append(alignment)
                except (ValueError, IndexError):
                    continue
            if parsing_data and line.startswith("="):
                parsing_data = False
        
        self.alignment_data = alignments
        print(f"Parsed {len(alignments)} alignment points for 3D reconstruction")
        
    def enable_processing_buttons(self):
        """Enable processing buttons when data is loaded."""
        self.process_button.config(state="normal")
        if self.alignment_data:
            self.analyze_button.config(state="normal")
        self.process_status_label.config(text="Ready for 3D reconstruction", foreground="blue")
        
    def generate_3d_reconstruction(self):
        """Generate 3D reconstruction based on selected method."""
        if not self.alignment_data and self.depth_map_data is None:
            messagebox.showwarning("No Data", "Please load data first")
            return
            
        try:
            self.process_button.config(state="disabled")
            self.progress_var.set(0)
            start_time = time.time()
            
            method = self.reconstruction_method.get()
            
            if method == "point_cloud":
                if self.alignment_data:
                    self.generate_point_cloud()
                else:
                    self.generate_point_cloud_from_png()
            elif method == "delaunay_mesh":
                if self.alignment_data:
                    self.generate_delaunay_mesh()
                else:
                    self.generate_point_cloud_from_png()
                    self.generate_delaunay_mesh()
            elif method == "png_heightfield":
                self.generate_png_heightfield_mesh()
            elif method == "hybrid":
                self.generate_hybrid_reconstruction()
            elif method == "multi_scale":
                self.generate_multi_scale_reconstruction()
            
            processing_time = time.time() - start_time
            
            self.process_status_label.config(
                text=f"✅ 3D reconstruction complete ({processing_time:.2f}s)", 
                foreground="green"
            )
            self.export_button.config(state="normal")
            if OPEN3D_AVAILABLE:
                self.open3d_button.config(state="normal")
            
            self.update_statistics()
            self.visualize_3d_result()
            
            messagebox.showinfo("Success", f"3D reconstruction completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            messagebox.showerror("Error", f"3D reconstruction failed: {str(e)}")
        finally:
            self.process_button.config(state="normal")
            self.progress_var.set(100)
            
    def generate_point_cloud_from_png(self):
        """Generate point cloud from PNG depth map."""
        if self.depth_map_data is None:
            raise ValueError("No PNG depth map loaded")
        
        self.progress_var.set(10)
        self.root.update_idletasks()
        
        height, width = self.depth_map_data.shape
        points = []
        qualities = []
        
        # Sample points from PNG depth map
        sample_rate = max(1, height // self.mesh_resolution.get())
        
        for y in range(0, height, sample_rate):
            for x in range(0, width, sample_rate):
                depth_value = self.depth_map_data[y, x]
                
                # Skip if depth is below threshold
                if depth_value < self.quality_threshold.get():
                    continue
                
                # Convert pixel coordinates to normalized 3D coordinates
                norm_x = (x / width) * 2.0 - 1.0
                norm_y = (y / height) * 2.0 - 1.0
                norm_z = depth_value
                
                points.append([norm_x, norm_y, norm_z])
                qualities.append(depth_value)  # Use depth as quality
        
        self.progress_var.set(50)
        self.root.update_idletasks()
        
        if not points:
            raise ValueError("No points generated from PNG depth map. Try lowering quality threshold.")
        
        self.point_cloud_3d = np.array(points)
        self.quality_weights = np.array(qualities)
        
        # Apply clustering if enabled
        if self.clustering_enabled.get() and SKLEARN_AVAILABLE and len(points) > 10:
            self.apply_clustering()
        
        # Remove noise if enabled
        if self.noise_removal.get():
            self.remove_noise_points()
        
        self.progress_var.set(90)
        self.root.update_idletasks()
        
        # Generate colors based on depth
        self.generate_depth_colors()
        
        print(f"Generated point cloud from PNG with {len(self.point_cloud_3d)} points")
        
    def generate_png_heightfield_mesh(self):
        """Generate mesh from PNG as heightfield."""
        if self.depth_map_data is None:
            raise ValueError("No PNG depth map loaded")
        
        self.progress_var.set(20)
        self.root.update_idletasks()
        
        height, width = self.depth_map_data.shape
        resolution = min(self.mesh_resolution.get(), min(height, width))
        
        vertices = []
        faces = []
        
        # Downsample if needed
        step_x = width // resolution
        step_y = height // resolution
        
        # Generate vertices
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                depth_value = self.depth_map_data[y, x]
                
                # Convert to normalized coordinates
                norm_x = (x / width) * 2.0 - 1.0
                norm_y = (y / height) * 2.0 - 1.0
                norm_z = depth_value
                
                vertices.append([norm_x, norm_y, norm_z])
        
        self.progress_var.set(50)
        self.root.update_idletasks()
        
        # Generate faces (triangles)
        cols = width // step_x
        rows = height // step_y
        
        for row in range(rows - 1):
            for col in range(cols - 1):
                # Current quad vertices
                v1 = row * cols + col
                v2 = row * cols + (col + 1)
                v3 = (row + 1) * cols + col
                v4 = (row + 1) * cols + (col + 1)
                
                # Check bounds
                if v4 < len(vertices):
                    # Create two triangles per quad
                    faces.append([v1, v2, v3])
                    faces.append([v2, v4, v3])
        
        self.mesh_vertices = np.array(vertices)
        self.mesh_faces = np.array(faces)
        
        # Also create point cloud
        self.point_cloud_3d = self.mesh_vertices.copy()
        self.quality_weights = self.mesh_vertices[:, 2]  # Use Z as quality
        
        self.progress_var.set(80)
        self.root.update_idletasks()
        
        # Apply smoothing if requested
        if self.smoothing_iterations.get() > 0:
            self.smooth_mesh()
        
        # Generate colors
        self.generate_depth_colors()
        
        print(f"Generated PNG heightfield mesh with {len(vertices)} vertices and {len(faces)} faces")
        
    def generate_point_cloud(self):
        """Generate 3D point cloud from alignment data."""
        self.progress_var.set(10)
        self.root.update_idletasks()
        
        points = []
        qualities = []
        conditions = []
        
        for alignment in self.alignment_data:
            # Extract 3D coordinates from G_refl components
            point = [
                alignment['G_refl_x'],
                alignment['G_refl_y'],
                alignment['G_refl_z']
            ]
            
            # Calculate quality score
            quality = 1.0 / (alignment['dist_boundary'] + 1e-6)
            
            # Filter by quality threshold
            if quality >= self.quality_threshold.get():
                points.append(point)
                qualities.append(quality)
                conditions.append(alignment['condition'])
        
        self.progress_var.set(50)
        self.root.update_idletasks()
        
        self.point_cloud_3d = np.array(points)
        self.quality_weights = np.array(qualities)
        
        # Apply clustering if enabled and sklearn is available
        if self.clustering_enabled.get() and SKLEARN_AVAILABLE and len(points) > 10:
            self.apply_clustering()
        
        # Remove noise if enabled
        if self.noise_removal.get():
            self.remove_noise_points()
        
        self.progress_var.set(90)
        self.root.update_idletasks()
        
        # Generate condition-based colors
        self.generate_condition_colors(conditions)
        
        print(f"Generated point cloud with {len(self.point_cloud_3d)} points")
        
    def generate_depth_colors(self):
        """Generate colors based on depth values."""
        if self.point_cloud_3d is None:
            return
        
        # Use Z coordinate for coloring
        z_values = self.point_cloud_3d[:, 2]
        
        # Normalize to [0, 1]
        if z_values.max() > z_values.min():
            normalized_z = (z_values - z_values.min()) / (z_values.max() - z_values.min())
        else:
            normalized_z = np.ones_like(z_values) * 0.5
        
        # Create colors using a colormap
        colormap = plt.cm.viridis
        self.condition_colors = colormap(normalized_z)[:, :3]  # RGB only
        
    def generate_delaunay_mesh(self):
        """Generate mesh using Delaunay triangulation."""
        if not hasattr(self, 'point_cloud_3d') or self.point_cloud_3d is None:
            if self.alignment_data:
                self.generate_point_cloud()
            else:
                self.generate_point_cloud_from_png()
        
        self.progress_var.set(30)
        self.root.update_idletasks()
        
        if len(self.point_cloud_3d) < 4:
            raise ValueError("Need at least 4 points for Delaunay triangulation")
        
        # Create Delaunay triangulation
        tri = Delaunay(self.point_cloud_3d)
        
        self.mesh_vertices = self.point_cloud_3d
        self.mesh_faces = tri.simplices
        
        self.progress_var.set(70)
        self.root.update_idletasks()
        
        # Apply smoothing if requested
        if self.smoothing_iterations.get() > 0:
            self.smooth_mesh()
        
        print(f"Generated Delaunay mesh with {len(self.mesh_vertices)} vertices and {len(self.mesh_faces)} faces")
        
    def generate_hybrid_reconstruction(self):
        """Generate both point cloud and mesh."""
        if self.alignment_data:
            self.generate_point_cloud()
        else:
            self.generate_point_cloud_from_png()
        
        self.progress_var.set(50)
        self.root.update_idletasks()
        
        if self.depth_map_data is not None:
            # Use PNG heightfield for mesh
            self.generate_png_heightfield_mesh()
        else:
            # Use Delaunay triangulation
            self.generate_delaunay_mesh()
        
        print("Generated hybrid reconstruction (points + mesh)")
        
    def generate_multi_scale_reconstruction(self):
        """Generate multi-scale reconstruction with different levels of detail."""
        original_threshold = self.quality_threshold.get()
        
        if self.alignment_data:
            # High quality level
            self.quality_threshold.set(0.5)
            self.generate_point_cloud()
            high_quality_points = self.point_cloud_3d.copy()
            
            # Medium quality level
            self.quality_threshold.set(0.1)
            self.generate_point_cloud()
            medium_quality_points = self.point_cloud_3d.copy()
            
            # Combine for multi-scale
            self.point_cloud_3d = np.vstack([high_quality_points, medium_quality_points])
            
            # Restore original threshold
            self.quality_threshold.set(original_threshold)
            
            # Generate mesh from combined points
            self.generate_delaunay_mesh()
        else:
            # Use PNG at different resolutions
            original_resolution = self.mesh_resolution.get()
            
            # High resolution
            self.mesh_resolution.set(256)
            self.generate_point_cloud_from_png()
            high_res_points = self.point_cloud_3d.copy()
            
            # Lower resolution
            self.mesh_resolution.set(128)
            self.generate_point_cloud_from_png()
            low_res_points = self.point_cloud_3d.copy()
            
            # Combine
            self.point_cloud_3d = np.vstack([high_res_points, low_res_points])
            
            # Restore resolution
            self.mesh_resolution.set(original_resolution)
            
            # Generate heightfield mesh
            self.generate_png_heightfield_mesh()
        
        print(f"Generated multi-scale reconstruction with {len(self.point_cloud_3d)} total points")
        
    def apply_clustering(self):
        """Apply DBSCAN clustering to remove outliers."""
        if not SKLEARN_AVAILABLE:
            print("Clustering skipped - scikit-learn not available")
            return
            
        if self.point_cloud_3d is None or len(self.point_cloud_3d) < 10:
            return
        
        clustering = DBSCAN(eps=self.cluster_eps.get(), min_samples=5)
        cluster_labels = clustering.fit_predict(self.point_cloud_3d)
        
        # Keep only points in the largest cluster
        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
        if len(unique_labels) > 0:
            largest_cluster = unique_labels[np.argmax(counts)]
            mask = cluster_labels == largest_cluster
            
            self.point_cloud_3d = self.point_cloud_3d[mask]
            if self.quality_weights is not None:
                self.quality_weights = self.quality_weights[mask]
        
        print(f"Clustering retained {len(self.point_cloud_3d)} points")
        
    def remove_noise_points(self):
        """Remove statistical outliers from point cloud."""
        if self.point_cloud_3d is None or len(self.point_cloud_3d) < 10:
            return
        
        # Calculate mean and std for each dimension
        mean_point = np.mean(self.point_cloud_3d, axis=0)
        std_point = np.std(self.point_cloud_3d, axis=0)
        
        # Remove points beyond 2 standard deviations
        distances = np.linalg.norm(self.point_cloud_3d - mean_point, axis=1)
        threshold = 2.0 * np.linalg.norm(std_point)
        
        mask = distances <= threshold
        self.point_cloud_3d = self.point_cloud_3d[mask]
        if self.quality_weights is not None:
            self.quality_weights = self.quality_weights[mask]
        
        print(f"Noise removal retained {len(self.point_cloud_3d)} points")
        
    def generate_condition_colors(self, conditions):
        """Generate colors based on projection conditions."""
        color_map = {
            'XY': [1.0, 0.0, 0.0],  # Red
            'XZ': [0.0, 1.0, 0.0],  # Green
            'YZ': [0.0, 0.0, 1.0]   # Blue
        }
        
        self.condition_colors = []
        for condition in conditions:
            if condition and len(self.condition_colors) < len(self.point_cloud_3d):
                for key in color_map:
                    if key in condition:
                        self.condition_colors.append(color_map[key])
                        break
                else:
                    self.condition_colors.append([0.5, 0.5, 0.5])  # Gray for unknown
        
        # Pad colors if needed
        while len(self.condition_colors) < len(self.point_cloud_3d):
            self.condition_colors.append([0.5, 0.5, 0.5])
        
        self.condition_colors = np.array(self.condition_colors[:len(self.point_cloud_3d)])
        
    def smooth_mesh(self):
        """Apply Laplacian smoothing to mesh."""
        if self.mesh_vertices is None or self.mesh_faces is None:
            return
        
        iterations = self.smoothing_iterations.get()
        smoothed_vertices = self.mesh_vertices.copy()
        
        for _ in range(iterations):
            new_vertices = smoothed_vertices.copy()
            
            # For each vertex, average with neighbors
            for i, vertex in enumerate(smoothed_vertices):
                neighbors = []
                
                # Find neighboring vertices through faces
                for face in self.mesh_faces:
                    if i in face:
                        neighbors.extend([smoothed_vertices[j] for j in face if j != i])
                
                if neighbors:
                    new_vertices[i] = np.mean(neighbors, axis=0)
            
            smoothed_vertices = new_vertices
        
        self.mesh_vertices = smoothed_vertices
        
    def visualize_3d_result(self):
        """Visualize the 3D reconstruction result."""
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d')
        
        if self.point_cloud_3d is not None and len(self.point_cloud_3d) > 0:
            # Determine coloring
            colormap = self.colormap_3d.get()
            
            if colormap == "quality" and self.quality_weights is not None:
                colors = self.quality_weights
                scatter = ax.scatter(self.point_cloud_3d[:, 0], 
                                   self.point_cloud_3d[:, 1], 
                                   self.point_cloud_3d[:, 2],
                                   c=colors, cmap='viridis', 
                                   s=self.point_size.get()**2,
                                   alpha=self.transparency.get())
                self.fig.colorbar(scatter, ax=ax, label='Quality Score')
                
            elif colormap == "condition" and self.condition_colors is not None:
                ax.scatter(self.point_cloud_3d[:, 0], 
                          self.point_cloud_3d[:, 1], 
                          self.point_cloud_3d[:, 2],
                          c=self.condition_colors, 
                          s=self.point_size.get()**2,
                          alpha=self.transparency.get())
                          
            elif colormap == "depth":
                colors = self.point_cloud_3d[:, 2]  # Use Z coordinate for depth
                scatter = ax.scatter(self.point_cloud_3d[:, 0], 
                                   self.point_cloud_3d[:, 1], 
                                   self.point_cloud_3d[:, 2],
                                   c=colors, cmap='coolwarm', 
                                   s=self.point_size.get()**2,
                                   alpha=self.transparency.get())
                self.fig.colorbar(scatter, ax=ax, label='Depth (Z)')
                
            else:
                ax.scatter(self.point_cloud_3d[:, 0], 
                          self.point_cloud_3d[:, 1], 
                          self.point_cloud_3d[:, 2],
                          c=self.point_cloud_3d[:, 2], cmap=colormap, 
                          s=self.point_size.get()**2,
                          alpha=self.transparency.get())
        
        # Add mesh if available
        if (self.mesh_vertices is not None and self.mesh_faces is not None and 
            self.reconstruction_method.get() in ["delaunay_mesh", "png_heightfield", "hybrid"]):
            
            # Draw mesh wireframe (limited for performance)
            max_faces_to_show = min(500, len(self.mesh_faces))
            for i, face in enumerate(self.mesh_faces[:max_faces_to_show]):
                if len(face) >= 3:
                    try:
                        triangle = self.mesh_vertices[face[:3]]
                        ax.plot([triangle[0,0], triangle[1,0]], 
                               [triangle[0,1], triangle[1,1]], 
                               [triangle[0,2], triangle[1,2]], 'c-', alpha=0.3, linewidth=0.5)
                        ax.plot([triangle[1,0], triangle[2,0]], 
                               [triangle[1,1], triangle[2,1]], 
                               [triangle[1,2], triangle[2,2]], 'c-', alpha=0.3, linewidth=0.5)
                        ax.plot([triangle[2,0], triangle[0,0]], 
                               [triangle[2,1], triangle[0,1]], 
                               [triangle[2,2], triangle[0,2]], 'c-', alpha=0.3, linewidth=0.5)
                    except IndexError:
                        continue
        
        ax.set_xlabel('X (G_refl_x)')
        ax.set_ylabel('Y (G_refl_y)')
        ax.set_zlabel('Z (G_refl_z)')
        
        # Add method info to title
        method_name = self.reconstruction_method.get().replace("_", " ").title()
        data_source = "PNG" if self.depth_map_data is not None else "Alignment"
        ax.set_title(f'G-S 3D Reconstruction - {method_name} ({data_source})')
        
        # Equal aspect ratio
        max_range = 1.0
        if self.point_cloud_3d is not None:
            max_range = np.max(np.abs(self.point_cloud_3d)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        self.canvas.draw()
        
    def analyze_quality_distribution(self):
        """Analyze and visualize quality distribution of alignments."""
        if not self.alignment_data:
            messagebox.showwarning("No Data", "Please load alignment data first")
            return
        
        # Create analysis window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("G-S Quality Analysis")
        analysis_window.geometry("800x600")
        
        fig = Figure(figsize=(10, 8), dpi=100)
        canvas = FigureCanvasTkAgg(fig, analysis_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Extract quality metrics
        distances_boundary = [a['dist_boundary'] for a in self.alignment_data]
        distances_origin = [a['dist_origin'] for a in self.alignment_data]
        qualities = [1.0 / (d + 1e-6) for d in distances_boundary]
        
        # Create subplots
        ax1 = fig.add_subplot(221)
        ax1.hist(distances_boundary, bins=50, alpha=0.7, color='red')
        ax1.set_xlabel('Distance from Boundary')
        ax1.set_ylabel('Count')
        ax1.set_title('Boundary Distance Distribution')
        ax1.set_yscale('log')
        
        ax2 = fig.add_subplot(222)
        ax2.hist(qualities, bins=50, alpha=0.7, color='blue')
        ax2.set_xlabel('Quality Score (1/dist_boundary)')
        ax2.set_ylabel('Count')
        ax2.set_title('Quality Score Distribution')
        ax2.set_yscale('log')
        
        ax3 = fig.add_subplot(223)
        ax3.scatter(distances_boundary, distances_origin, alpha=0.5, s=1)
        ax3.set_xlabel('Distance from Boundary')
        ax3.set_ylabel('Distance from Origin')
        ax3.set_title('Boundary vs Origin Distance')
        
        ax4 = fig.add_subplot(224)
        # Condition breakdown
        conditions = {}
        for a in self.alignment_data:
            cond = a['condition'] if a['condition'] else 'Unknown'
            conditions[cond] = conditions.get(cond, 0) + 1
        
        if conditions:
            labels = [k.replace(' PROJECTION', '').replace(', NUIT RADIUS = ', '\nR=') 
                     for k in conditions.keys()]
            ax4.pie(conditions.values(), labels=labels, autopct='%1.1f%%')
            ax4.set_title('Alignments by Condition')
        
        fig.tight_layout()
        canvas.draw()
        
        # Add statistics text
        stats_text = f"""
Quality Analysis Summary:
Total Alignments: {len(self.alignment_data)}
Perfect Alignments (dist_boundary < 0.001): {sum(1 for d in distances_boundary if d < 0.001)}
High Quality (quality > 100): {sum(1 for q in qualities if q > 100)}
Mean Quality: {np.mean(qualities):.2f}
Median Quality: {np.median(qualities):.2f}
Quality Range: [{np.min(qualities):.2f}, {np.max(qualities):.2f}]
"""
        
        text_label = tk.Label(analysis_window, text=stats_text, 
                             font=("Courier", 10), justify=tk.LEFT)
        text_label.pack(pady=10)
        
    def update_statistics(self):
        """Update statistics display."""
        stats_text = "3D Reconstruction Statistics\n"
        stats_text += "=" * 30 + "\n"
        
        if self.point_cloud_3d is not None:
            stats_text += f"Point Cloud: {len(self.point_cloud_3d)} points\n"
            stats_text += f"Bounds X: [{self.point_cloud_3d[:, 0].min():.3f}, {self.point_cloud_3d[:, 0].max():.3f}]\n"
            stats_text += f"Bounds Y: [{self.point_cloud_3d[:, 1].min():.3f}, {self.point_cloud_3d[:, 1].max():.3f}]\n"
            stats_text += f"Bounds Z: [{self.point_cloud_3d[:, 2].min():.3f}, {self.point_cloud_3d[:, 2].max():.3f}]\n"
        
        if self.mesh_vertices is not None and self.mesh_faces is not None:
            stats_text += f"Mesh: {len(self.mesh_vertices)} vertices, {len(self.mesh_faces)} faces\n"
        
        if self.quality_weights is not None:
            stats_text += f"Quality range: [{self.quality_weights.min():.2f}, {self.quality_weights.max():.2f}]\n"
        
        if self.depth_map_data is not None:
            stats_text += f"PNG Depth Map: {self.depth_map_data.shape}\n"
            stats_text += f"PNG Depth Range: [{self.depth_map_data.min():.3f}, {self.depth_map_data.max():.3f}]\n"
        
        stats_text += f"Method: {self.reconstruction_method.get()}\n"
        stats_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.stats_label.config(text=stats_text)
        
    def export_3d_model(self):
        """Export 3D model in selected format."""
        if self.point_cloud_3d is None:
            messagebox.showwarning("No Data", "Generate 3D reconstruction first")
            return
        
        format_type = self.export_format.get()
        
        filename = filedialog.asksaveasfilename(
            title=f"Save 3D Model As {format_type}",
            defaultextension=f".{format_type.lower()}",
            filetypes=[(f"{format_type} files", f"*.{format_type.lower()}"), 
                      ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            if format_type == "PLY":
                self.export_ply(filename)
            elif format_type == "OBJ":
                self.export_obj(filename)
            elif format_type == "STL":
                self.export_stl(filename)
            elif format_type == "JSON":
                self.export_json(filename)
            elif format_type == "CSV":
                self.export_csv(filename)
            
            messagebox.showinfo("Export Success", f"3D model saved as {format_type}: {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export 3D model: {str(e)}")
            
    def export_ply(self, filename):
        """Export as PLY format."""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment G-S 3D Reconstruction from PNG depth map\n")
            f.write(f"element vertex {len(self.point_cloud_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if self.include_colors.get() and self.condition_colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            if self.quality_weights is not None:
                f.write("property float quality\n")
            
            if (self.mesh_faces is not None and 
                self.reconstruction_method.get() in ["delaunay_mesh", "png_heightfield", "hybrid"]):
                f.write(f"element face {len(self.mesh_faces)}\n")
                f.write("property list uchar int vertex_indices\n")
            
            f.write("end_header\n")
            
            # Write vertices
            for i, vertex in enumerate(self.point_cloud_3d):
                line = f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}"
                
                if self.include_colors.get() and self.condition_colors is not None:
                    color = (self.condition_colors[i] * 255).astype(int)
                    line += f" {color[0]} {color[1]} {color[2]}"
                
                if self.quality_weights is not None:
                    line += f" {self.quality_weights[i]:.6f}"
                
                f.write(line + "\n")
            
            # Write faces if available
            if (self.mesh_faces is not None and 
                self.reconstruction_method.get() in ["delaunay_mesh", "png_heightfield", "hybrid"]):
                for face in self.mesh_faces:
                    if len(face) >= 3:
                        f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
                    
    def export_obj(self, filename):
        """Export as OBJ format."""
        with open(filename, 'w') as f:
            f.write("# G-S 3D Reconstruction from PNG by Angledcrystals\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Method: {self.reconstruction_method.get()}\n")
            f.write(f"# Source: PNG depth map\n\n")
            
            # Write vertices
            for vertex in self.point_cloud_3d:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces if available
            if (self.mesh_faces is not None and 
                self.reconstruction_method.get() in ["delaunay_mesh", "png_heightfield", "hybrid"]):
                f.write("\n")
                for face in self.mesh_faces:
                    if len(face) >= 3:
                        # OBJ uses 1-based indexing
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                    
    def export_stl(self, filename):
        """Export as STL format (ASCII)."""
        if self.mesh_faces is None:
            raise ValueError("STL export requires mesh data. Use 'png_heightfield' or 'delaunay_mesh' method.")
        
        with open(filename, 'w') as f:
            f.write(f"solid GS_3D_Reconstruction_PNG\n")
            
            for face in self.mesh_faces:
                if len(face) >= 3 and all(idx < len(self.mesh_vertices) for idx in face[:3]):
                    # Calculate normal vector
                    try:
                        v1 = self.mesh_vertices[face[1]] - self.mesh_vertices[face[0]]
                        v2 = self.mesh_vertices[face[2]] - self.mesh_vertices[face[0]]
                        normal = np.cross(v1, v2)
                        normal_mag = np.linalg.norm(normal)
                        if normal_mag > 0:
                            normal = normal / normal_mag
                        else:
                            normal = np.array([0, 0, 1])
                        
                        f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                        f.write(f"    outer loop\n")
                        
                        for vertex_idx in face[:3]:
                            vertex = self.mesh_vertices[vertex_idx]
                            f.write(f"      vertex {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                        
                        f.write(f"    endloop\n")
                        f.write(f"  endfacet\n")
                    except (IndexError, ValueError):
                        continue
            
            f.write(f"endsolid GS_3D_Reconstruction_PNG\n")
            
    def export_json(self, filename):
        """Export as JSON format with metadata."""
        data = {
            "metadata": {
                "generator": "G-S 3D Reconstructor PNG Support by Angledcrystals",
                "version": "1.2",
                "date": datetime.now().isoformat(),
                "method": self.reconstruction_method.get(),
                "total_alignments": len(self.alignment_data) if self.alignment_data else 0,
                "point_count": len(self.point_cloud_3d) if self.point_cloud_3d is not None else 0,
                "has_png_depth_map": self.depth_map_data is not None,
                "png_shape": self.depth_map_data.shape if self.depth_map_data is not None else None,
                "png_processing": {
                    "scale_factor": self.depth_scale_factor.get(),
                    "inverted": self.depth_invert.get(),
                    "normalized": self.depth_normalize.get()
                },
                "sklearn_available": SKLEARN_AVAILABLE,
                "open3d_available": OPEN3D_AVAILABLE
            },
            "points": self.point_cloud_3d.tolist() if self.point_cloud_3d is not None else [],
            "qualities": self.quality_weights.tolist() if self.quality_weights is not None else [],
            "colors": self.condition_colors.tolist() if self.condition_colors is not None else []
        }
        
        if self.mesh_vertices is not None and self.mesh_faces is not None:
            data["mesh"] = {
                "vertices": self.mesh_vertices.tolist(),
                "faces": self.mesh_faces.tolist()
            }
        
        if self.depth_map_data is not None:
            # Include depth map statistics
            data["depth_map_stats"] = {
                "shape": self.depth_map_data.shape,
                "min_depth": float(self.depth_map_data.min()),
                "max_depth": float(self.depth_map_data.max()),
                "mean_depth": float(self.depth_map_data.mean()),
                "std_depth": float(self.depth_map_data.std())
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def export_csv(self, filename):
        """Export as CSV format."""
        with open(filename, 'w') as f:
            # Write header
            header = "x,y,z"
            if self.quality_weights is not None:
                header += ",quality"
            if self.condition_colors is not None:
                header += ",color_r,color_g,color_b"
            header += ",source"
            f.write(header + "\n")
            
            # Write data
            for i, point in enumerate(self.point_cloud_3d):
                line = f"{point[0]:.6f},{point[1]:.6f},{point[2]:.6f}"
                
                if self.quality_weights is not None:
                    line += f",{self.quality_weights[i]:.6f}"
                
                if self.condition_colors is not None:
                    color = self.condition_colors[i]
                    line += f",{color[0]:.6f},{color[1]:.6f},{color[2]:.6f}"
                
                # Add source information
                source = "PNG" if self.depth_map_data is not None else "ALIGNMENT"
                line += f",{source}"
                
                f.write(line + "\n")
            
    def open_in_open3d(self):
        """Open current reconstruction in Open3D viewer."""
        if not OPEN3D_AVAILABLE:
            messagebox.showerror("Error", "Open3D not available")
            return
        
        if self.point_cloud_3d is None:
            messagebox.showwarning("No Data", "Generate 3D reconstruction first")
            return
        
        try:
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud_3d)
            
            if self.condition_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(self.condition_colors)
            
            # Estimate normals for better visualization
            pcd.estimate_normals()
            
            geometries = [pcd]
            
            # Add mesh if available
            if (self.mesh_vertices is not None and self.mesh_faces is not None and 
                self.reconstruction_method.get() in ["delaunay_mesh", "png_heightfield", "hybrid"]):
                
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(self.mesh_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(self.mesh_faces)
                mesh.paint_uniform_color([0.7, 0.7, 0.7])
                mesh.compute_vertex_normals()
                
                geometries.append(mesh)
            
            # Visualize
            window_name = "G-S 3D Reconstruction - PNG Support - Open3D Viewer"
            o3d.visualization.draw_geometries(geometries, window_name=window_name)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open in Open3D: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GS3DReconstructorGUI(root)
    root.mainloop()
