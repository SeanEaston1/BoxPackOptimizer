import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import permutations

# Box Optimizer Class
class BoxOptimizer:
    def __init__(self, truck_dims, box_dims):
        self.truck_dims = np.array(truck_dims)
        self.original_boxes = np.array(box_dims)
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

        for box, original_idx in zip(boxes, box_indices):
            pos, rot = self.try_place_box(box, placed_boxes)
            if pos is not None:
                placed_boxes.append({
                    'dimensions': rot,
                    'position': pos,
                    'original_index': original_idx + 1  # Add 1 for 1-based indexing
                })
            else:
                return None  # If any box can't be placed, arrangement is invalid

        return placed_boxes

    def optimize(self):
        """Find optimal arrangement of boxes with maximum volume utilization"""
        boxes = self.original_boxes.copy()
        n_boxes = len(boxes)
        max_volume = 0
        best_arrangement = None

        # Try all possible combinations of boxes
        for r in range(1, n_boxes + 1):
            for indices in permutations(range(n_boxes), r):
                selected_boxes = boxes[list(indices)]
                arrangement = self.try_arrangement(selected_boxes, indices)

                if arrangement:
                    total_volume = sum(self.get_volume(box['dimensions']) 
                                    for box in arrangement)

                    if total_volume > max_volume:
                        max_volume = total_volume
                        best_arrangement = arrangement
                        self.best_utilization = max_volume / self.get_volume(self.truck_dims)

        self.best_arrangement = best_arrangement or []
        return self.best_arrangement, self.best_utilization

# Visualization Functions
def generate_box_colors(num_boxes):
    """Generate distinct colors for boxes"""
    colors = [
        'red', 'green', 'blue', 'yellow', 'magenta', 'cyan',
        'brown', 'orange', 'purple', 'pink'
    ]
    return colors[:num_boxes]

def create_3d_visualization(truck_dims, box_arrangements):
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

    colors = generate_box_colors(len(box_arrangements))
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
        for idx, box in enumerate(box_arrangements):
            pos = box['position']
            dims = box['dimensions']
            original_idx = box.get('original_index', idx + 1)  # Use original index if available

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

            if ax == ax1:  # Only add to legend for the first view
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=colors[idx], 
                                                    label=f'Box {original_idx} ({dims[0]}x{dims[1]}x{dims[2]})'))

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

    # Box dimensions
    st.sidebar.subheader("Box Dimensions")
    num_boxes = st.sidebar.number_input("Number of Boxes", min_value=1, max_value=10, value=6)

    box_dimensions = []
    for i in range(num_boxes):
        st.sidebar.markdown(f"**Box {i+1}**")
        col1, col2, col3 = st.sidebar.columns(3)

        with col1:
            length = st.number_input(f"Length {i+1}", min_value=1, value=3, key=f"length_{i}")
        with col2:
            width = st.number_input(f"Width {i+1}", min_value=1, value=2, key=f"width_{i}")
        with col3:
            height = st.number_input(f"Height {i+1}", min_value=1, value=2, key=f"height_{i}")

        box_dimensions.append([length, width, height])

    # Optimize button
    if st.sidebar.button("Optimize Arrangement"):
        # Create optimizer instance
        optimizer = BoxOptimizer(
            truck_dims=[truck_length, truck_width, truck_height],
            box_dims=box_dimensions
        )

        # Run optimization
        with st.spinner("Optimizing box arrangement..."):
            arrangement, utilization = optimizer.optimize()

        # Display results
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("3D Visualization")
            fig = create_3d_visualization(
                [truck_length, truck_width, truck_height],
                arrangement
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
            for idx, box in enumerate(arrangement):
                st.markdown(f"""
                    **Box {box['original_index']}:**
                    - Dimensions: {box['dimensions']}
                    - Position: {box['position']}
                """)

    # Instructions
    if 'help' not in st.session_state:
        st.info("""
            **How to use:**
            1. Enter truck dimensions in the sidebar
            2. Specify the number of boxes and their dimensions
            3. Click 'Optimize Arrangement' to see the results
            4. View the 3D visualization and arrangement details

            The optimizer will attempt to find the best arrangement while considering
            possible rotations of the boxes.
        """)

if __name__ == "__main__":
    main()
