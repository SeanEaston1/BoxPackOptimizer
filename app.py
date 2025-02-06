import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import permutations

# Box Optimizer Class
class BoxOptimizer:
    def __init__(self, truck_dims, box_dims, box_weights, max_weight):
        self.truck_dims = np.array(truck_dims)
        self.original_boxes = np.array(box_dims)
        self.box_weights = np.array(box_weights)
        self.max_weight = max_weight
        self.best_arrangement = []
        self.best_utilization = 0
        self.all_valid_arrangements = []

    def get_volume(self, dims):
        return np.prod(dims)

    def can_fit(self, box, position, placed_boxes):
        """Check if box can fit at position without overlapping"""
        box_end = position + box

        # Check if box fits within truck dimensions
        if np.any(box_end > self.truck_dims):
            return False

        # Check overlap with other boxes
        for placed_box in placed_boxes:
            placed_pos = placed_box['position']
            placed_dims = placed_box['dimensions']
            placed_end = placed_pos + placed_dims

            # Check for overlap in all dimensions
            if (np.all(position < placed_end) and 
                np.all(placed_pos < box_end)):
                return False

        return True

    def get_possible_rotations(self, box):
        """Get all possible rotations of a box"""
        return np.array(list(set(permutations(box))))

    def try_place_box(self, box, placed_boxes):
        """Try to place a box in all possible positions and rotations"""
        best_pos = None
        best_rot = None
        min_waste = float('inf')

        rotations = self.get_possible_rotations(box)

        for rot_box in rotations:
            # Try positions starting from bottom
            for x in range(int(self.truck_dims[0] - rot_box[0] + 1)):
                for y in range(int(self.truck_dims[1] - rot_box[1] + 1)):
                    for z in range(int(self.truck_dims[2] - rot_box[2] + 1)):
                        pos = np.array([x, y, z])
                        if self.can_fit(rot_box, pos, placed_boxes):
                            # Calculate wasted space (distance from origin)
                            waste = np.sum(pos)
                            if waste < min_waste:
                                min_waste = waste
                                best_pos = pos
                                best_rot = rot_box

        return best_pos, best_rot

    def try_arrangement(self, boxes, box_indices):
        """Try to arrange a specific combination of boxes"""
        placed_boxes = []
        total_weight = 0

        for box, original_idx in zip(boxes, box_indices):
            pos, rot = self.try_place_box(box, placed_boxes)
            if pos is not None:
                box_weight = self.box_weights[original_idx]
                total_weight += box_weight
                if total_weight > self.max_weight:
                    return None #arrangement exceeds weight limit

                placed_boxes.append({
                    'dimensions': rot,
                    'position': pos,
                    'original_index': original_idx + 1  # Add 1 for 1-based indexing
                })
            else:
                return None  # If any box can't be placed, arrangement is invalid

        return placed_boxes

    def optimize(self):
        """Find optimal arrangement of boxes with maximum volume utilization and weight consideration"""
        boxes = self.original_boxes.copy()
        n_boxes = len(boxes)
        max_volume = 0
        best_arrangement = None

        # Sort boxes by weight (heaviest first)
        sorted_indices = np.argsort(self.box_weights)[::-1]
        boxes = boxes[sorted_indices]
        sorted_weights = np.array(self.box_weights)[sorted_indices]

        # Try arrangements starting with heavier boxes
        for r in range(1, n_boxes + 1):
            for indices in permutations(range(n_boxes), r):
                selected_boxes = boxes[list(indices)]
                selected_indices = sorted_indices[list(indices)]
                arrangement = self.try_arrangement(selected_boxes, selected_indices)

                if arrangement:
                    total_volume = sum(self.get_volume(box['dimensions']) 
                                    for box in arrangement)

                    # Check if boxes are stacked correctly by weight
                    valid_weight_arrangement = True
                    for i, box1 in enumerate(arrangement):
                        box1_weight = self.box_weights[box1['original_index']-1]
                        box1_bottom = box1['position'][2]
                        
                        for j, box2 in enumerate(arrangement):
                            if i != j:
                                box2_weight = self.box_weights[box2['original_index']-1]
                                box2_bottom = box2['position'][2]
                                
                                # If a heavier box is above a lighter box
                                if box2_bottom > box1_bottom and box2_weight > box1_weight:
                                    valid_weight_arrangement = False
                                    break
                        
                        if not valid_weight_arrangement:
                            break

                    if valid_weight_arrangement and total_volume > max_volume:
                        max_volume = total_volume
                        best_arrangement = arrangement
                        self.best_utilization = max_volume / self.get_volume(self.truck_dims)

        self.best_arrangement = best_arrangement or []
        return self.best_arrangement, self.best_utilization

# Visualization Functions
def generate_box_colors(num_boxes, scheme):
    """Generate distinct colors for boxes based on selected scheme"""
    if scheme == "Pastel":
        colors = plt.cm.Pastel1(np.linspace(0, 1, num_boxes))
    elif scheme == "Bold":
        colors = plt.cm.Set1(np.linspace(0, 1, num_boxes))
    elif scheme == "Grayscale":
        colors = plt.cm.gray(np.linspace(0, 1, num_boxes))
    else:  # Default
        colors = [
            'red', 'green', 'blue', 'yellow', 'magenta', 'cyan',
            'brown', 'orange', 'purple', 'pink'
        ]
    return colors[:num_boxes]


def create_3d_visualization(truck_dims, box_arrangements, colors):
    """Create multiple views visualization using Matplotlib"""
    fig = plt.figure(figsize=(20, 12))

    # Create subplots for different views
    ax1 = fig.add_subplot(231, projection='3d')  # 3D view
    ax2 = fig.add_subplot(232, projection='3d')  # Front view
    ax3 = fig.add_subplot(233, projection='3d')  # Back view
    ax4 = fig.add_subplot(235, projection='3d')  # Top view
    ax5 = fig.add_subplot(236, projection='3d')  # Bottom view

    views = [
        (ax1, 'Default 3D View', (45, 45)),
        (ax2, 'Front View', (0, 0)),
        (ax3, 'Back View', (0, 180)),
        (ax4, 'Top View', (90, -90)),
        (ax5, 'Bottom View', (-90, -90))
    ]


    legend_elements = []

    for view_info in views:
        ax, title, (elev, azim) = view_info

        # Plot truck outline
        truck_vertices = np.array([
            [0, 0, 0], [truck_dims[0], 0, 0], [truck_dims[0], truck_dims[1], 0],
            [0, truck_dims[1], 0], [0, 0, truck_dims[2]], 
            [truck_dims[0], 0, truck_dims[2]],
            [truck_dims[0], truck_dims[1], truck_dims[2]], 
            [0, truck_dims[1], truck_dims[2]]
        ])

        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        for edge in edges:
            ax.plot3D(
                [truck_vertices[edge[0]][0], truck_vertices[edge[1]][0]],
                [truck_vertices[edge[0]][1], truck_vertices[edge[1]][1]],
                [truck_vertices[edge[0]][2], truck_vertices[edge[1]][2]],
                'black', linewidth=1
            )

        # Plot boxes
        placed_boxes_indices = set()  # Keep track of actually placed boxes
        for idx, box in enumerate(box_arrangements):
            pos = box['position']
            dims = box['dimensions']
            original_idx = box.get('original_index', idx + 1)
            placed_boxes_indices.add(original_idx)

            # Define vertices of the box
            vertices = np.array([
                [pos[0], pos[1], pos[2]],
                [pos[0] + dims[0], pos[1], pos[2]],
                [pos[0] + dims[0], pos[1] + dims[1], pos[2]],
                [pos[0], pos[1] + dims[1], pos[2]],
                [pos[0], pos[1], pos[2] + dims[2]],
                [pos[0] + dims[0], pos[1], pos[2] + dims[2]],
                [pos[0] + dims[0], pos[1] + dims[1], pos[2] + dims[2]],
                [pos[0], pos[1] + dims[1], pos[2] + dims[2]]
            ])

            # Define faces of the box
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[3], vertices[7], vertices[4]]
            ]

            # Plot box
            poly = Poly3DCollection(faces, alpha=0.3)
            poly.set_facecolor(colors[idx])
            ax.add_collection3d(poly)

            # Add to legend only for the first view and only for placed boxes
            if ax == ax1:
                weight = box_weights[original_idx-1]
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=colors[idx], 
                                                  label=f'Box {original_idx} ({dims[0]}x{dims[1]}x{dims[2]}) - {weight}kg'))

        # Set view angle
        ax.view_init(elev=elev, azim=azim)

        # Set labels and title
        ax.set_xlabel('Length')
        ax.set_ylabel('Width')
        ax.set_zlabel('Height')
        ax.set_title(title)

        # Set axis limits
        ax.set_xlim([0, truck_dims[0]])
        ax.set_ylim([0, truck_dims[1]])
        ax.set_zlim([0, truck_dims[2]])

    # Add legend to the figure
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.52))

    # Adjust layout
    plt.tight_layout()

    return fig

def calculate_statistics(truck_dims, box_arrangements):
    """Calculate space utilization statistics"""
    truck_volume = np.prod(truck_dims)
    utilized_volume = sum(np.prod(box['dimensions']) for box in box_arrangements)
    utilization_percentage = (utilized_volume / truck_volume) * 100

    return {
        'truck_volume': truck_volume,
        'utilized_volume': utilized_volume,
        'utilization_percentage': utilization_percentage
    }

# Streamlit App
def main():
    st.set_page_config(page_title="3D Box Arrangement Optimizer", layout="wide")

    st.title("3D Box Arrangement Optimizer")

    # Sidebar inputs
    st.sidebar.header("Input Dimensions")

    # Truck dimensions
    st.sidebar.subheader("Truck Dimensions")
    truck_length = st.sidebar.number_input("Truck Length", min_value=1, value=5)
    truck_width = st.sidebar.number_input("Truck Width", min_value=1, value=5)
    truck_height = st.sidebar.number_input("Truck Height", min_value=1, value=3)

    # Box dimensions and weights
    st.sidebar.subheader("Box Configuration")
    num_boxes = st.sidebar.number_input("Number of Boxes", min_value=1, max_value=10, value=6)

    # Weight limit
    max_weight = st.sidebar.number_input("Maximum Total Weight (kg)", min_value=0.0, value=1000.0)

    # Color customization
    color_scheme = st.sidebar.selectbox(
        "Color Scheme",
        ["Default", "Pastel", "Bold", "Grayscale"]
    )

    box_dimensions = []
    box_weights = []
    for i in range(num_boxes):
        st.sidebar.markdown(f"**Box {i+1}**")
        col1, col2, col3, col4 = st.sidebar.columns(4)

        with col1:
            length = st.number_input(f"Length {i+1}", min_value=1, value=3, key=f"length_{i}")
        with col2:
            width = st.number_input(f"Width {i+1}", min_value=1, value=2, key=f"width_{i}")
        with col3:
            height = st.number_input(f"Height {i+1}", min_value=1, value=2, key=f"height_{i}")
        with col4:
            weight = st.number_input(f"Weight {i+1} (kg)", min_value=0.0, value=10.0, key=f"weight_{i}")

        box_dimensions.append([length, width, height])
        box_weights.append(weight)

    # Optimize button
    if st.sidebar.button("Optimize Arrangement"):
        # Create optimizer instance
        optimizer = BoxOptimizer(
            truck_dims=[truck_length, truck_width, truck_height],
            box_dims=box_dimensions,
            box_weights=box_weights,
            max_weight=max_weight
        )

        # Run optimization
        with st.spinner("Optimizing box arrangement..."):
            arrangement, utilization = optimizer.optimize()

        # Display results
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("3D Visualization")
            colors = generate_box_colors(len(arrangement), color_scheme)
            fig = create_3d_visualization(
                [truck_length, truck_width, truck_height],
                arrangement,
                colors
            )
            st.pyplot(fig)

        with col2:
            st.subheader("Statistics")
            stats = calculate_statistics(
                [truck_length, truck_width, truck_height],
                arrangement
            )

            st.metric("Truck Volume", f"{stats['truck_volume']:.2f} cubic units")
            st.metric("Utilized Volume", f"{stats['utilized_volume']:.2f} cubic units")
            st.metric("Space Utilization", f"{stats['utilization_percentage']:.2f}%")

            st.subheader("Arrangement Details")
            arrangement_data = []
            total_weight = 0

            for box in arrangement:
                box_weight = box_weights[box['original_index']-1]
                total_weight += box_weight
                arrangement_data.append({
                    'box_number': box['original_index'],
                    'dimensions': box['dimensions'].tolist(),
                    'position': box['position'].tolist(),
                    'weight': box_weight
                })
                st.markdown(f"""
                    **Box {box['original_index']}:**
                    - Dimensions: {box['dimensions']}
                    - Position: {box['position']}
                    - Weight: {box_weight} kg
                """)

            st.metric("Total Weight", f"{total_weight:.2f} kg")

            # Export functionality
            if st.button("Export Arrangement"):
                import json
                import base64

                export_data = {
                    'truck_dimensions': [truck_length, truck_width, truck_height],
                    'arrangement': arrangement_data,
                    'statistics': stats
                }

                json_str = json.dumps(export_data, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'data:application/json;base64,{b64}'
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="box_arrangement.json",
                    mime="application/json"
                )

    # Instructions
    if 'help' not in st.session_state:
        st.info("""
            **How to use:**
            1. Enter truck dimensions in the sidebar
            2. Specify the number of boxes and their dimensions and weights
            3. Set a maximum total weight limit.
            4. Choose a color scheme.
            5. Click 'Optimize Arrangement' to see the results
            6. View the 3D visualization and arrangement details
            7. Click 'Export Arrangement' to download the results as a JSON file.

            The optimizer will attempt to find the best arrangement while considering
            possible rotations of the boxes and respecting the weight limit.
        """)

if __name__ == "__main__":
    main()
