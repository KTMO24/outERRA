"""
outERRA Framework
by Travis Michael O’Dell 2025
All rights reserved.
"""

import asyncio
import json
import os
import time
import subprocess
import threading
import queue
import inspect
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from jupyter_client import KernelManager

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# -------------------- Backend Components --------------------

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Event and State Management
class ExecutionState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    WAITING = "waiting"

@dataclass
class ExecutionEvent:
    type: str
    data: Dict[str, Any]
    timestamp: datetime = datetime.now()

class EventBus:
    """Central event management system with WebSocket support."""
    def __init__(self, socketio: SocketIO):
        self.subscribers: Dict[str, List[Callable[[ExecutionEvent], None]]] = {}
        self.event_queue = asyncio.Queue()
        self.running = True
        self.socketio = socketio
        asyncio.create_task(self._event_processor())

    def subscribe(self, event_type: str, callback: Callable[[ExecutionEvent], None]):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[ExecutionEvent], None]):
        self.subscribers.setdefault("all", []).append(callback)

    async def publish(self, event: ExecutionEvent):
        await self.event_queue.put(event)

    async def _event_processor(self):
        while self.running:
            event = await self.event_queue.get()
            # Handle specific event type subscribers
            if event.type in self.subscribers:
                for callback in self.subscribers[event.type]:
                    try:
                        callback(event)
                    except Exception as e:
                        logging.error(f"Error in event callback: {str(e)}")
            # Handle 'all' subscribers
            if "all" in self.subscribers:
                for callback in self.subscribers["all"]:
                    try:
                        callback(event)
                    except Exception as e:
                        logging.error(f"Error in 'all' event callback: {str(e)}")

# JupyterRobot Class
class JupyterRobot:
    def __init__(self, workspace_dir: str, gemini_api_key: str, event_bus: EventBus):
        """
        Initialize the Jupyter notebook robot controller

        Args:
            workspace_dir: Base directory for task management
            gemini_api_key: Google Gemini API key for AI operations
            event_bus: EventBus instance for event management
        """
        self.workspace_dir = Path(workspace_dir)
        self.gemini_api_key = gemini_api_key
        self.event_bus = event_bus
        self.current_task = None
        self.task_queue = []

        # Create workspace structure
        self.tasks_dir = self.workspace_dir / "tasks"
        self.dependencies_dir = self.workspace_dir / "dependencies"
        self.state_file = self.workspace_dir / "robot_state.json"

        self._init_workspace()

    def _init_workspace(self):
        """Create necessary directories and state file"""
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.dependencies_dir.mkdir(parents=True, exist_ok=True)

        if not self.state_file.exists():
            self._save_state({
                "active_tasks": [],
                "completed_tasks": [],
                "task_dependencies": {}
            })

    def _save_state(self, state: dict):
        """Save current state to JSON file"""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> dict:
        """Load current state from JSON file"""
        with open(self.state_file, 'r') as f:
            return json.load(f)

    def create_task(self, task_name: str, notebook_path: str, dependencies: List[str] = None) -> str:
        """
        Create a new notebook task with specified dependencies

        Args:
            task_name: Unique identifier for the task
            notebook_path: Path to the Jupyter notebook
            dependencies: List of task names this task depends on
        """
        task_id = f"{task_name}_{int(time.time())}"
        task_dir = self.tasks_dir / task_id
        task_dir.mkdir(parents=True)

        # Copy notebook to task directory
        target_notebook = task_dir / "notebook.ipynb"
        with open(notebook_path, 'r') as source, open(target_notebook, 'w') as target:
            notebook_content = json.load(source)
            json.dump(notebook_content, target)

        # Create task metadata
        task_meta = {
            "task_id": task_id,
            "name": task_name,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "dependencies": dependencies or []
        }

        with open(task_dir / "metadata.json", 'w') as f:
            json.dump(task_meta, f, indent=2)

        # Update global state
        state = self._load_state()
        state["active_tasks"].append(task_id)
        if dependencies:
            state["task_dependencies"][task_id] = dependencies
        self._save_state(state)

        # Publish event
        asyncio.create_task(self.event_bus.publish(ExecutionEvent(
            type="task_created",
            data={"task_id": task_id, "task_name": task_name}
        )))

        logging.info(f"Task created: {task_id}")
        return task_id

    def execute_notebook(self, task_id: str):
        """Execute a notebook using nbconvert"""
        task_dir = self.tasks_dir / task_id
        notebook_path = task_dir / "notebook.ipynb"

        try:
            # Execute notebook using nbconvert
            cmd = [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(notebook_path)
            ]
            subprocess.run(cmd, check=True)

            # Update task status
            self._update_task_status(task_id, "completed")

            # Publish event
            asyncio.create_task(self.event_bus.publish(ExecutionEvent(
                type="task_completed",
                data={"task_id": task_id}
            )))

            logging.info(f"Task completed: {task_id}")

        except subprocess.CalledProcessError as e:
            self._update_task_status(task_id, "failed")

            # Publish event
            asyncio.create_task(self.event_bus.publish(ExecutionEvent(
                type="task_failed",
                data={"task_id": task_id, "error": str(e)}
            )))

            logging.error(f"Task failed: {task_id} with error: {str(e)}")
            raise Exception(f"Failed to execute notebook: {str(e)}")

    def query_gemini(self, prompt: str) -> str:
        """
        Query Google Gemini API using subprocess to call curl

        Args:
            prompt: Text prompt for Gemini
        """
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        api_key = self.gemini_api_key

        # Prepare the request using curl
        curl_cmd = [
            'curl', '-X', 'POST',
            f'{url}?key={api_key}',
            '-H', 'Content-Type: application/json',
            '-d', json.dumps({
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            })
        ]

        try:
            result = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
            response = json.loads(result.stdout)
            generated_text = response['candidates'][0]['content']['parts'][0]['text']
            logging.info(f"Gemini response: {generated_text}")
            return generated_text
        except Exception as e:
            logging.error(f"Failed to query Gemini API: {str(e)}")
            raise Exception(f"Failed to query Gemini API: {str(e)}")

    def _update_task_status(self, task_id: str, status: str):
        """Update task status in metadata"""
        task_dir = self.tasks_dir / task_id
        meta_file = task_dir / "metadata.json"

        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        metadata["status"] = status
        metadata["updated_at"] = datetime.now().isoformat()

        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_task_status(self, task_id: str) -> dict:
        """Get current status and metadata for a task"""
        task_dir = self.tasks_dir / task_id
        meta_file = task_dir / "metadata.json"

        with open(meta_file, 'r') as f:
            return json.load(f)

    def process_queue(self):
        """Process tasks in queue based on dependencies"""
        state = self._load_state()

        for task_id in state["active_tasks"]:
            if task_id in state["task_dependencies"]:
                deps = state["task_dependencies"][task_id]
                # Check if all dependencies are completed
                deps_completed = all(
                    self.get_task_status(dep)["status"] == "completed"
                    for dep in deps
                )
                if deps_completed:
                    self.execute_notebook(task_id)
            else:
                # No dependencies, execute immediately
                self.execute_notebook(task_id)

        self._clean_completed_tasks()

    def _clean_completed_tasks(self):
        """Move completed tasks to completed_tasks list"""
        state = self._load_state()
        completed = [
            task_id for task_id in state["active_tasks"]
            if self.get_task_status(task_id)["status"] == "completed"
        ]

        for task_id in completed:
            state["active_tasks"].remove(task_id)
            state["completed_tasks"].append(task_id)
            if task_id in state["task_dependencies"]:
                del state["task_dependencies"][task_id]

        self._save_state(state)

# Execution Controller
class ExecutionController:
    """Main control interface for the Jupyter Robot system"""
    def __init__(self, robot: JupyterRobot, event_bus: EventBus):
        self.robot = robot
        self.event_bus = event_bus
        self.state = ExecutionState.IDLE
        self.current_task = None
        self.task_queue = queue.PriorityQueue()
        self.lock = threading.Lock()

        # Register event handlers
        self.event_bus.subscribe("task_completed", self._handle_task_complete)
        self.event_bus.subscribe("task_failed", self._handle_task_error)
        # Additional event subscriptions can be added here

        # Start the task processing loop
        threading.Thread(target=self._task_processor, daemon=True).start()

    def execute_task(self, task_id: str, priority: int = 1):
        """Schedule a task for execution with priority"""
        self.task_queue.put((priority, task_id))
        asyncio.run_coroutine_threadsafe(
            self.event_bus.publish(ExecutionEvent(
                type="task_scheduled",
                data={"task_id": task_id, "priority": priority}
            )),
            asyncio.get_event_loop()
        )

    def _task_processor(self):
        """Continuously process tasks from the queue"""
        while True:
            priority, task_id = self.task_queue.get()
            with self.lock:
                if self.state == ExecutionState.IDLE:
                    self.state = ExecutionState.RUNNING
                    self.current_task = task_id
                    threading.Thread(target=self._run_task, args=(task_id,), daemon=True).start()

    def _run_task(self, task_id: str):
        """Run the specified task"""
        try:
            self.robot.execute_notebook(task_id)
        except Exception as e:
            logging.error(f"Task {task_id} failed with error: {e}")
        finally:
            with self.lock:
                self.state = ExecutionState.IDLE
                self.current_task = None

    def _handle_task_complete(self, event: ExecutionEvent):
        """Handle task completion event"""
        task_id = event.data["task_id"]
        if task_id == self.current_task:
            self.state = ExecutionState.IDLE
            self.current_task = None
            # Automatically process the next task
            threading.Thread(target=self._task_processor, daemon=True).start()

    def _handle_task_error(self, event: ExecutionEvent):
        """Handle task execution error"""
        task_id = event.data["task_id"]
        error = event.data.get("error", "Unknown error")
        logging.error(f"Task {task_id} failed with error: {error}")
        if task_id == self.current_task:
            self.state = ExecutionState.IDLE
            self.current_task = None
            # Optionally implement retry logic here
            threading.Thread(target=self._task_processor, daemon=True).start()

# NotebookController Class
class NotebookController:
    """Controls all Jupyter notebook operations"""
    def __init__(self):
        self.kernel_manager = KernelManager()
        self.active_notebooks: Dict[str, dict] = {}
        self.execution_queue = queue.Queue()
        self.operation_history: List[dict] = []

    def create_notebook(self, notebook_id: str, cells: List[Dict] = None) -> dict:
        """Create a new notebook with optional initial cells"""
        notebook = new_notebook()

        if cells:
            for cell_content in cells:
                if cell_content["type"] == "code":
                    cell = new_code_cell(cell_content["content"])
                else:
                    cell = new_markdown_cell(cell_content["content"])

                if cell_content.get("metadata"):
                    cell.metadata.update(cell_content["metadata"])

                notebook.cells.append(cell)

        self.active_notebooks[notebook_id] = {
            'notebook': notebook,
            'kernel': None,
            'execution_count': 0
        }

        self._log_operation("create", notebook_id)
        return {'notebook_id': notebook_id, 'cell_count': len(notebook.cells)}

    def start_kernel(self, notebook_id: str):
        """Start a kernel for the notebook"""
        if notebook_id not in self.active_notebooks:
            raise ValueError(f"Notebook {notebook_id} not found")

        self.kernel_manager.start_kernel()
        self.active_notebooks[notebook_id]['kernel'] = self.kernel_manager

        return {'status': 'kernel_started', 'notebook_id': notebook_id}

    def execute_cell(self, notebook_id: str, cell_index: int) -> dict:
        """Execute a specific cell in the notebook"""
        if notebook_id not in self.active_notebooks:
            raise ValueError(f"Notebook {notebook_id} not found")

        notebook_info = self.active_notebooks[notebook_id]
        notebook = notebook_info['notebook']

        if cell_index >= len(notebook.cells):
            raise ValueError(f"Cell index {cell_index} out of range")

        cell = notebook.cells[cell_index]
        if cell.cell_type != 'code':
            return {'status': 'skipped', 'reason': 'not_code_cell'}

        kernel = notebook_info['kernel']
        if not kernel:
            self.start_kernel(notebook_id)
            kernel = self.active_notebooks[notebook_id]['kernel']

        msg_id = kernel.client.execute(cell.source)
        notebook_info['execution_count'] += 1

        self._log_operation("execute", notebook_id, {'cell_index': cell_index})
        return {
            'status': 'executed',
            'msg_id': msg_id,
            'execution_count': notebook_info['execution_count']
        }

    def insert_cell(self, notebook_id: str, cell_content: Dict, position: int = None) -> dict:
        """Insert a new cell at the specified position"""
        if notebook_id not in self.active_notebooks:
            raise ValueError(f"Notebook {notebook_id} not found")

        notebook = self.active_notebooks[notebook_id]['notebook']

        if cell_content["type"] == "code":
            cell = new_code_cell(cell_content["content"])
        else:
            cell = new_markdown_cell(cell_content["content"])

        if cell_content.get("metadata"):
            cell.metadata.update(cell_content["metadata"])

        if position is None:
            position = len(notebook.cells)

        notebook.cells.insert(position, cell)
        self._log_operation("insert_cell", notebook_id, {'position': position})

        return {'cell_index': position, 'notebook_id': notebook_id}

    def update_cell(self, notebook_id: str, cell_index: int, cell_content: Dict) -> dict:
        """Update content of an existing cell"""
        if notebook_id not in self.active_notebooks:
            raise ValueError(f"Notebook {notebook_id} not found")

        notebook = self.active_notebooks[notebook_id]['notebook']

        if cell_index >= len(notebook.cells):
            raise ValueError(f"Cell index {cell_index} out of range")

        cell = notebook.cells[cell_index]
        cell.source = cell_content["content"]

        if cell_content.get("metadata"):
            cell.metadata.update(cell_content["metadata"])

        self._log_operation("update_cell", notebook_id, {'cell_index': cell_index})
        return {'status': 'updated', 'cell_index': cell_index}

    def delete_cell(self, notebook_id: str, cell_index: int) -> dict:
        """Delete a cell from the notebook"""
        if notebook_id not in self.active_notebooks:
            raise ValueError(f"Notebook {notebook_id} not found")

        notebook = self.active_notebooks[notebook_id]['notebook']

        if cell_index >= len(notebook.cells):
            raise ValueError(f"Cell index {cell_index} out of range")

        del notebook.cells[cell_index]
        self._log_operation("delete_cell", notebook_id, {'cell_index': cell_index})

        return {'status': 'deleted', 'cell_index': cell_index}

    def save_notebook(self, notebook_id: str, filepath: str) -> dict:
        """Save the notebook to a file"""
        if notebook_id not in self.active_notebooks:
            raise ValueError(f"Notebook {notebook_id} not found")

        notebook = self.active_notebooks[notebook_id]['notebook']

        with open(filepath, 'w') as f:
            nbformat.write(notebook, f)

        self._log_operation("save_notebook", notebook_id, {'filepath': filepath})
        return {'status': 'saved', 'filepath': filepath}

    def _log_operation(self, operation: str, notebook_id: str, details: dict = None):
        """Log notebook operations for history tracking"""
        log_entry = {
            'operation': operation,
            'notebook_id': notebook_id,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.operation_history.append(log_entry)

# InterfaceManager Class
class InterfaceManager:
    """Manages interaction surfaces for the execution system"""
    def __init__(self, controller: ExecutionController, notebook_controller: NotebookController):
        self.controller = controller
        self.notebook_controller = notebook_controller
        self.active_interfaces: Dict[str, Any] = {}
        self.interface_states: Dict[str, dict] = {}

    async def register_interface(self, interface_id: str, interface_type: str):
        """Register a new interface for interaction"""
        if interface_id in self.active_interfaces:
            raise ValueError(f"Interface {interface_id} already registered")

        interface_state = {
            "type": interface_type,
            "status": "active",
            "last_update": datetime.now().isoformat()
        }

        self.interface_states[interface_id] = interface_state
        self.active_interfaces[interface_id] = self._create_interface(interface_type)

        return self.active_interfaces[interface_id]

    def _create_interface(self, interface_type: str) -> dict:
        """Create interface methods based on type"""
        if interface_type == "notebook":
            return {
                "execute": self.controller.execute_task,
                "get_status": self.notebook_controller.get_task_status,
                "query_ai": self.controller.robot.query_gemini,
                "create_notebook": self.notebook_controller.create_notebook,
                "insert_cell": self.notebook_controller.insert_cell,
                "update_cell": self.notebook_controller.update_cell,
                "delete_cell": self.notebook_controller.delete_cell,
                "save_notebook": self.notebook_controller.save_notebook,
                "execute_cell": self.notebook_controller.execute_cell
            }
        elif interface_type == "monitoring":
            return {
                "get_state": lambda: self.controller.state.value,
                "get_queue": lambda: list(self.controller.task_queue.queue),
                "get_current": lambda: self.controller.current_task
            }
        else:
            raise ValueError(f"Unknown interface type: {interface_type}")

    async def call_interface(self, interface_id: str, method: str, *args, **kwargs):
        """Call a method on a registered interface"""
        if interface_id not in self.active_interfaces:
            raise ValueError(f"Interface {interface_id} not registered")

        interface = self.active_interfaces[interface_id]
        if method not in interface:
            raise ValueError(f"Method {method} not available on interface {interface_id}")

        return await interface[method](*args, **kwargs)

# ApplicationController Class
class ApplicationController:
    """Main application controller that ties everything together"""
    def __init__(self, workspace_dir: str, gemini_api_key: str, socketio: SocketIO):
        self.event_bus = EventBus(socketio)
        self.robot = JupyterRobot(workspace_dir, gemini_api_key, self.event_bus)
        self.notebook_controller = NotebookController()
        self.executor = ExecutionController(self.robot, self.event_bus)
        self.interface_manager = InterfaceManager(self.executor, self.notebook_controller)
        self.socketio = socketio

    def start(self):
        """Initialize the application"""
        # Register default interfaces
        asyncio.run_coroutine_threadsafe(
            self.interface_manager.register_interface(
                "main_notebook", "notebook"
            ),
            asyncio.get_event_loop()
        )
        asyncio.run_coroutine_threadsafe(
            self.interface_manager.register_interface(
                "system_monitor", "monitoring"
            ),
            asyncio.get_event_loop()
        )

        # Set up event handling for broadcasting events
        self.event_bus.subscribe_all(self._broadcast_event)

    def _broadcast_event(self, event: ExecutionEvent):
        """Handle broadcasting events to WebSocket connections"""
        self.socketio.emit('event', {
            "type": event.type,
            "data": event.data
        })

# -------------------- Outerra Framework Components --------------------

# Base Class Definitions
class Berry:
    def __init__(self, shrub):
        self.shrub = shrub
        self.name = "base_berry"
        self.description = "Base Berry"
        self.phylum = None
        self.growth = None

    def run(self, job):
        print(f"[{self.name}] Received job: {job}")
        return {"status": "success", "message": f"Job processed by {self.name}"}

class Shrub:
    def __init__(self, garden, name="default_shrub", description="Default Shrub Description"):
        self.garden = garden
        self.name = name
        self.description = description
        self.berries = {}
        self.queue = queue.Queue()
        self.leaves = {}
        self.phylum = None
        self.growth = None
        self.pot = None
        self.microclimate = "normal"

    def initialize(self):
        self.load_berries()
        self.register_leaves()
        print(f"[{self.name}] Initialized (Phylum: {self.phylum}, Growth: {self.growth})")

    def load_berries(self):
        pass  # Load berries dynamically if required

    def register_leaves(self):
        for name, method in inspect.getmembers(self):
            if hasattr(method, 'is_leaf') and method.is_leaf:
                self.leaves[name] = method
                print(f"[{self.name}] Registered leaf: {name}")

    def add_job(self, job):
        self.queue.put(job)

    def process_queue(self):
        while True:
            job = self.queue.get()
            berry_name = job.get("berry")
            if berry_name:
                berry = self.berries.get(berry_name)
                if berry:
                    result = berry.run(job)
                    print(result)
                else:
                    print(f"Berry '{berry_name}' not found in shrub '{self.name}'")
            else:
                print("No berry specified in the job")

    def expose_leaf(self, is_async=False):
        def decorator(func):
            func.is_leaf = True
            func.is_async = is_async
            return func
        return decorator

    def is_active(self):
        return True

# Trunk Class (Already defined above)
class Trunk:
    def __init__(self, garden):
        self.garden = garden
        self.name = "trunk"
        self.description = "Central trunk of the garden"
        self.berries = {}
        self.phylum = "core"
        self.growth = "base"
        self.load_berries()

    def load_berries(self):
        self.berries['logic_generator'] = LogicGeneratorBerry(self)

    def run(self, job):
        berry_name = job.get("berry")
        if berry_name in self.berries:
             return self.berries[berry_name].run(job)
        else:
             return {"status": "error", "message": f"Berry '{berry_name}' not found in the Trunk."}


# Modular Logic for Self-Generation
class LogicGeneratorBerry(Berry):
    def __init__(self, shrub):
        super().__init__(shrub)
        self.name = "logic_generator"
        self.description = "Generates new logic based on input"

    def run(self, job):
        job_type = job.get("type")
        if job_type == "generate_new_module":
            # Logic to generate new modules or logic dynamically
            module_name = f"generated_module_{random.randint(1, 100)}"
            new_module = SelfGeneratingModule(self.shrub.garden, name=module_name)
            self.shrub.garden.add_shrub(new_module)
            return {"status": "success", "new_module": module_name}
        return {"status": "error", "message": "Invalid job type"}

# Garden Initialization and Dynamic Module Loading
class Garden:
    def __init__(self, config: Dict[str, Any]):
        self.shrubs = {}
        self.global_queue = queue.Queue()
        self.trunk = Trunk(self)
        self.config = config
        self.load_shrubs()

    def load_shrubs(self):
        for shrub_config in self.config.get("shrubs", []):
            shrub_class = self.get_shrub_class(shrub_config['name'])
            if shrub_class:
                shrub = shrub_class(
                    self,
                    name=shrub_config.get('name', 'default_shrub'),
                    description=shrub_config.get('description', 'Default Description')
                )
                self.shrubs[shrub.name] = shrub
                threading.Thread(target=shrub.process_queue, daemon=True).start()
                shrub.initialize()

    def get_shrub_class(self, name: str):
        # Placeholder for loading Shrub classes dynamically based on configuration
        if name == "SelfGeneratingModule":
            return SelfGeneratingModule
        elif name == "LogicGenerator":
            return LogicGeneratorBerry  # Adjust based on actual class hierarchy
        return Shrub  # Default to base Shrub

    def add_shrub(self, shrub: Shrub):
        self.shrubs[shrub.name] = shrub
        threading.Thread(target=shrub.process_queue, daemon=True).start()
        shrub.initialize()

    def get_shrub(self, name: str) -> Optional[Shrub]:
        return self.shrubs.get(name)

# Self-Generating Logic
class SelfGeneratingModule(Shrub):
    def __init__(self, garden: Garden, name: str = "self_generating_shrub", description: str = "Self-Generating Module"):
        super().__init__(garden, name, description)
        self.growth = "deep"

    def initialize(self):
        # Automatically generates new logic or Shrubs
        new_berry = LogicGeneratorBerry(self)
        self.berries[new_berry.name] = new_berry
        print(f"{self.name} initialized and ready for logic generation")

# Dynamic GUI with ipywidgets
class DynamicGUI:
    def __init__(self, garden: Garden):
        self.garden = garden
        self.output = widgets.Output()
        self.update_button = widgets.Button(description="Update Modules")
        self.update_button.on_click(self.update_modules)
        self.control_panel = widgets.VBox([self.update_button, self.output])
        display(self.control_panel)
        self.update_modules(None)  # Initial display

    def update_modules(self, b):
        with self.output:
            clear_output()
            for shrub in self.garden.shrubs.values():
                print(f"Module {shrub.name} is running with {len(shrub.berries)} berries.")

    def refresh(self):
        self.update_modules(None)

# -------------------- Flask Application --------------------

# Embedded HTML, CSS, and JavaScript for the frontend
# For simplicity, we embed the frontend as a multi-line string
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>outERRA Control Panel</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        #control-panel {
            border: 1px solid #ccc;
            padding: 20px;
            border-radius: 5px;
        }
        #status-area {
            margin-top: 20px;
            height: 300px;
            overflow-y: scroll;
            border: 1px solid #eee;
            padding: 10px;
        }
        .status-message {
            padding: 5px;
            border-bottom: 1px solid #eee;
        }
        #notebook-controls {
          margin-top: 20px;
          border: 1px solid #ccc;
          padding: 10px;
          border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="control-panel">
        <h2>outERRA Control Panel</h2>
        <button id <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>outERRA Control Panel</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        #control-panel {
            border: 1px solid #ccc;
            padding: 20px;
            border-radius: 5px;
        }
        #status-area {
            margin-top: 20px;
            height: 300px;
            overflow-y: scroll;
            border: 1px solid #eee;
            padding: 10px;
        }
        .status-message {
            padding: 5px;
            border-bottom: 1px solid #eee;
        }
        #notebook-controls {
          margin-top: 20px;
          border: 1px solid #ccc;
          padding: 10px;
          border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="control-panel">
        <h2>outERRA Control Panel</h2>
        <button id="create-task-btn">Create Task</button>
        <button id="execute-notebook-btn">Execute Notebook</button>
        <div id="status-area"></div>
    </div>
    <div id="notebook-controls">
        <h3>Notebook Operations</h3>
        <button id="create-notebook-btn">Create Notebook</button>
        <button id="insert-cell-btn">Insert Cell</button>
        <button id="update-cell-btn">Update Cell</button>
        <button id="delete-cell-btn">Delete Cell</button>
        <button id="execute-cell-btn">Execute Cell</button>
        <button id="save-notebook-btn">Save Notebook</button>
    </div>


    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js" integrity="sha512-UZ8CXuWX0UPYZPmk8TqsMd3F0vO3E4ZlzZh0pbupZHcS3S7Ht+YYIYdjYvjt5G+G/uufXCBYrbDsKkbj8FwCiw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script>
        // Initialize WebSocket connection using Socket.IO
        const socket = io();

        socket.on('connect', () => {
            console.log("WebSocket connection established.");
        });

        socket.on('event', (data) => {
            handleEvent(data);
        });

        socket.on('disconnect', () => {
            console.log("WebSocket connection closed.");
        });

        // Event handling
        function handleEvent(data) {
            const { type, data: eventData } = data;
            const statusArea = document.getElementById("status-area");

            let message = "";
            if (type === "task_created") {
                message = `Task Created: ${eventData.task_id}`;
            } else if (type === "task_completed") {
                message = `Task Completed: ${eventData.task_id}`;
            } else if (type === "task_failed") {
                message = `Task Failed: ${eventData.task_id} - ${eventData.error}`;
            } else if (type === "task_scheduled") {
                message = `Task Scheduled: ${eventData.task_id} with priority ${eventData.priority}`;
            }

            if (message) {
                const msgDiv = document.createElement("div");
                msgDiv.className = "status-message";
                msgDiv.textContent = message;
                statusArea.prepend(msgDiv);
            }
        }

        // Button handlers for task creation and execution
        document.getElementById("create-task-btn").addEventListener("click", async () => {
            const taskName = prompt("Enter Task Name:");
            if (!taskName) return;
            const notebookPath = prompt("Enter Notebook Path (relative to workspace):");
            if (!notebookPath) return;
            const dependenciesInput = prompt("Enter Dependencies (comma-separated task IDs):");
            const dependencies = dependenciesInput ? dependenciesInput.split(",").map(dep => dep.trim()) : [];

            const response = await fetch("/create_task/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ task_name: taskName, notebook_path: notebookPath, dependencies })
            });

            const result = await response.json();
            const statusArea = document.getElementById("status-area");
            if (response.ok) {
                statusArea.innerHTML += `<p>Task Created: ${result.task_id}</p>`;
            } else {
                statusArea.innerHTML += `<p style="color:red;">Error: ${result.error}</p>`;
            }
        });

        document.getElementById("execute-notebook-btn").addEventListener("click", async () => {
            const taskId = prompt("Enter Task ID to Execute:");
            if (!taskId) return;
            const priority = parseInt(prompt("Enter Priority (1=High, 2=Medium, 3=Low):"), 10) || 1;

            const response = await fetch("/execute_notebook/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ task_id: taskId, priority })
            });

            const result = await response.json();
            const statusArea = document.getElementById("status-area");
            if (response.ok) {
                statusArea.innerHTML += `<p>Execution Started for Task: ${result.task_id} with Priority: ${priority}</p>`;
            } else {
                statusArea.innerHTML += `<p style="color:red;">Error: ${result.error}</p>`;
            }
        });

       //Button Handlers for Notebook Operations
        document.getElementById("create-notebook-btn").addEventListener("click", async () => {
           const notebookId = prompt("Enter a Notebook ID:");
           if(!notebookId) return;
           const response = await fetch("/create_notebook/", {
                method: "POST",
                headers: {
                   "Content-Type": "application/json"
                },
                body: JSON.stringify({ notebook_id: notebookId })
            });
           const result = await response.json();
           const statusArea = document.getElementById("status-area");
            if (response.ok) {
                statusArea.innerHTML += `<p>Notebook Created: ${result.notebook_id} with ${result.cell_count} Cells </p>`;
            } else {
                statusArea.innerHTML += `<p style="color:red;">Error creating notebook: ${result.error}</p>`;
            }
        });

        document.getElementById("insert-cell-btn").addEventListener("click", async () => {
             const notebookId = prompt("Enter Notebook ID:");
            if (!notebookId) return;
            const cellType = prompt("Enter cell type ('code' or 'markdown'):");
            if (!cellType) return;
            const cellContent = prompt("Enter cell content:");
            if (!cellContent) return;
            const position = parseInt(prompt("Enter cell position (0 for beginning, leave blank for end):"), 10);

            const response = await fetch("/insert_cell/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    notebook_id: notebookId,
                    cell_content: { type: cellType, content: cellContent },
                    position: isNaN(position) ? null : position
                })
            });

            const result = await response.json();
            const statusArea = document.getElementById("status-area");
             if (response.ok) {
                statusArea.innerHTML += `<p>Cell inserted into ${result.notebook_id} at index ${result.cell_index}</p>`;
            } else {
                statusArea.innerHTML += `<p style="color:red;">Error inserting cell: ${result.error}</p>`;
            }
        });

        document.getElementById("update-cell-btn").addEventListener("click", async () => {
             const notebookId = prompt("Enter Notebook ID:");
            if (!notebookId) return;
            const cellIndex = parseInt(prompt("Enter cell index to update:"), 10);
             if(isNaN(cellIndex)) return;
             const cellType = prompt("Enter cell type ('code' or 'markdown'):");
            if (!cellType) return;
            const cellContent = prompt("Enter new cell content:");
            if (!cellContent) return;
            const response = await fetch("/update_cell/", {
                method: "POST",
                headers: {
                   "Content-Type": "application/json"
                },
                body: JSON.stringify({
                   notebook_id: notebookId,
                   cell_index: cellIndex,
                   cell_content: {type: cellType, content: cellContent}
                })
            });
             const result = await response.json();
            const statusArea = document.getElementById("status-area");
            if(response.ok){
                statusArea.innerHTML += `<p>Cell at index ${result.cell_index} in notebook ${notebookId} updated</p>`
            } else {
                statusArea.innerHTML += `<p style="color:red;">Error updating cell: ${result.error}</p>`
            }
        });

        document.getElementById("delete-cell-btn").addEventListener("click", async () => {
            const notebookId = prompt("Enter Notebook ID:");
            if (!notebookId) return;
            const cellIndex = parseInt(prompt("Enter cell index to delete:"), 10);
            if (isNaN(cellIndex)) return;

            const response = await fetch("/delete_cell/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ notebook_id: notebookId, cell_index: cellIndex })
            });

            const result = await response.json();
            const statusArea = document.getElementById("status-area");
              if(response.ok){
                 statusArea.innerHTML += `<p>Cell at index ${result.cell_index} in notebook ${notebookId} deleted</p>`
              }else {
                 statusArea.innerHTML += `<p style="color:red;">Error deleting cell: ${result.error}</p>`
              }
        });

       document.getElementById("execute-cell-btn").addEventListener("click", async () => {
            const notebookId = prompt("Enter Notebook ID:");
            if(!notebookId) return;
           const cellIndex = parseInt(prompt("Enter cell index to execute:"), 10);
            if(isNaN(cellIndex)) return;
             const response = await fetch("/execute_cell/", {
                 method: "POST",
                headers: {
                     "Content-Type": "application/json"
                 },
                 body: JSON.stringify({
                     notebook_id: notebookId,
                     cell_index: cellIndex
                 })
            });
             const result = await response.json();
             const statusArea = document.getElementById("status-area");
               if(response.ok){
                  statusArea.innerHTML += `<p> Cell at index ${result.cell_index} in notebook ${notebookId} executed. Message ID: ${result.msg_id}</p>`
               }else {
                   statusArea.innerHTML += `<p style="color:red;">Error executing cell: ${result.error}</p>`
               }
       });


        document.getElementById("save-notebook-btn").addEventListener("click", async () => {
            const notebookId = prompt("Enter Notebook ID:");
            if (!notebookId) return;
            const filepath = prompt("Enter the filepath to save notebook (.ipynb):");
            if(!filepath) return;

             const response = await fetch("/save_notebook/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
               body: JSON.stringify({notebook_id: notebookId, filepath: filepath})
            });
             const result = await response.json();
             const statusArea = document.getElementById("status-area");
            if(response.ok){
               statusArea.innerHTML += `<p>Notebook ${notebookId} saved to ${result.filepath}</p>`;
            }else {
                statusArea.innerHTML += `<p style="color:red;">Error saving notebook: ${result.error}</p>`;
            }
        });

        // Optionally, you can implement polling or additional real-time features here
    </script>
</body>
</html>
"""

# Initialize Flask app
app_flask = Flask(__name__)
CORS(app_flask)  # Enable CORS for all domains

# Initialize Flask-SocketIO
socketio = SocketIO(app_flask, cors_allowed_origins="*")

# Initialize ApplicationController
# Adjust 'workspace_dir' and 'gemini_api_key' as necessary
workspace_directory = "workspace"
gemini_api_key = "YOUR_GEMINI_API_KEY"  # Replace with your actual Gemini API key
app_controller = ApplicationController(workspace_dir=workspace_directory, gemini_api_key=gemini_api_key, socketio=socketio)

# Initialize Garden with Outerra configuration
outerra_config = {
    "shrubs": [
        {"name": "SelfGeneratingModule", "description": "Module that can generate new logic"}
    ]
}
garden = Garden(outerra_config)

# Initialize Dynamic GUI
gui = DynamicGUI(garden)

# Start ApplicationController
def start_application():
    app_controller.start()

application_thread = threading.Thread(target=start_application, daemon=True)
application_thread.start()

# API Endpoints

@app_flask.route("/create_task/", methods=["POST"])
def create_task_endpoint():
    """
    Create a new notebook execution task.
    Expects JSON body with:
    - task_name: str
    - notebook_path: str (relative to workspace_dir)
    - dependencies: list of task_ids (optional)
    """
    data = request.get_json()
    task_name = data.get("task_name")
    notebook_path = data.get("notebook_path")
    dependencies = data.get("dependencies", [])

    if not task_name or not notebook_path:
        return jsonify({"error": "task_name and notebook_path are required."}), 400

    notebook_full_path = Path(app_controller.robot.workspace_dir) / notebook_path
    if not notebook_full_path.exists():
        return jsonify({"error": f"Notebook path {notebook_path} does not exist."}), 400

    task_id = app_controller.robot.create_task(task_name, str(notebook_full_path), dependencies)
    return jsonify({"status": "Task created", "task_id": task_id}), 200

@app_flask.route("/execute_notebook/", methods=["POST"])
def execute_notebook_endpoint():
    """
    Execute a notebook task by its ID with a given priority.
    Expects JSON body with:
    - task_id: str
    - priority: int (optional, default=1)
    """
    data = request.get_json()
    task_id = data.get("task_id")
    priority = data.get("priority", 1)

    if not task_id:
        return jsonify({"error": "task_id is required."}), 400

    state = app_controller.robot._load_state()
    if task_id not in state["active_tasks"]:
        return jsonify({"error": f"Task ID {task_id} not found in active tasks."}), 400

    app_controller.executor.execute_task(task_id, priority)
    return jsonify({"status": "Execution started", "task_id": task_id}), 200

@app_flask.route("/task_status/<task_id>", methods=["GET"])
def get_task_status_endpoint(task_id):
    """
    Get the status of a specific task.
    """
    try:
        status = app_controller.robot.get_task_status(task_id)
        return jsonify(status), 200
    except FileNotFoundError:
        return jsonify({"error": f"Task ID {task_id} not found."}), 404

#------ Notebook API endpoints ------
@app_flask.route("/create_notebook/", methods=["POST"])
def create_notebook_endpoint():
   """
   Create a new notebook
    Expects JSON body with:
    - notebook_id: str
    """
   data = request.get_json()
   notebook_id = data.get("notebook_id")
   if not notebook_id:
       return jsonify({"error": "notebook_id is required"}), 400
   try:
       notebook_data = app_controller.notebook_controller.create_notebook(notebook_id)
       return jsonify(notebook_data), 200
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app_flask.route("/insert_cell/", methods=["POST"])
def insert_cell_endpoint():
   """
   Insert a cell into a notebook
   Expects JSON body with:
        - notebook_id: str
        - cell_content: { type: 'code' or 'markdown', content: str}
        - position: int (optional)
   """
   data = request.get_json()
   notebook_id = data.get("notebook_id")
   cell_content = data.get("cell_content")
   position = data.get("position")

   if not notebook_id or not cell_content or not cell_content.get("type") or not cell_content.get("content"):
        return jsonify({"error": "notebook_id, cell_content(type, content) are required"}), 400

   try:
       cell_data = app_controller.interface_manager.call_interface(
           "main_notebook",
           "insert_cell",
           notebook_id,
           cell_content,
           position
       )

       return jsonify(asyncio.run(cell_data)), 200
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app_flask.route("/update_cell/", methods=["POST"])
def update_cell_endpoint():
    """
    Update content of a cell
    Expects JSON body with:
        - notebook_id: str
        - cell_index: int
        - cell_content: {type: 'code' or 'markdown', content: str}
    """
    data = request.get_json()
    notebook_id = data.get("notebook_id")
    cell_index = data.get("cell_index")
    cell_content = data.get("cell_content")

    if not notebook_id or cell_index is None or not cell_content or not cell_content.get("type") or not cell_content.get("content"):
         return jsonify({"error": "notebook_id, cell_index, cell_content(type, content) are required."}), 400
    try:
        cell_data = app_controller.interface_manager.call_interface(
            "main_notebook",
            "update_cell",
            notebook_id,
            cell_index,
            cell_content
        )
        return jsonify(asyncio.run(cell_data)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app_flask.route("/delete_cell/", methods=["POST"])
def delete_cell_endpoint():
    """
    Delete a cell from a notebook
    Expects JSON body with:
        - notebook_id: str
        - cell_index: int
    """
    data = request.get_json()
    notebook_id = data.get("notebook_id")
    cell_index = data.get("cell_index")

    if not notebook_id or cell_index is None:
        return jsonify({"error": "notebook_id and cell_index are required"}), 400
    try:
        cell_data = app_controller.interface_manager.call_interface(
            "main_notebook",
            "delete_cell",
            notebook_id,
            cell_index
        )
        return jsonify(asyncio.run(cell_data)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app_flask.route("/execute_cell/", methods=["POST"])
def execute_cell_endpoint():
    """
    Execute a specific cell from a notebook
    Expects JSON body with:
        - notebook_id: str
        - cell_index: int
    """
    data = request.get_json()
    notebook_id = data.get("notebook_id")
    cell_index = data.get("cell_index")

    if not notebook_id or cell_index is None:
        return jsonify({"error": "notebook_id and cell_index are required"}), 400
    try:
        cell_data = app_controller.interface_manager.call_interface(
            "main_notebook",
            "execute_cell",
            notebook_id,
            cell_index
        )
        return jsonify(asyncio.run(cell_data)), 200
    except Exception as e:
       return jsonify({"error": str(e)}), 500

@app_flask.route("/save_notebook/", methods=["POST"])
def save_notebook_endpoint():
    """
    Save notebook to file system
    Expects JSON body with:
        - notebook_id: str
        - filepath: str
    """
    data = request.get_json()
    notebook_id = data.get("notebook_id")
    filepath = data.get("filepath")

    if not notebook_id or not filepath:
        return jsonify({"error": "notebook_id and filepath are required"}), 400
    try:
        notebook_data = app_controller.interface_manager.call_interface(
            "main_notebook",
            "save_notebook",
            notebook_id,
            filepath
        )
        return jsonify(asyncio.run(notebook_data)), 200
    except Exception as e:
       return jsonify({"error": str(e)}), 500

# WebSocket Events

@socketio.on('connect')
def handle_connect():
    logging.info(f"Client connected: {request.sid}")
    emit('message', {'data': 'Connected to outERRA WebSocket'})

@socketio.on('disconnect')
def handle_disconnect():
    logging.info(f"Client disconnected: {request.sid}")

# Serve Frontend HTML
@app_flask.route("/", methods=["GET"])
def get_frontend():
    return HTML(FRONTEND_HTML), 200

# -------------------- Python Command Inputs --------------------

def create_task(task_name: str, notebook_path: str, dependencies: List[str] = None) -> Dict[str, Any]:
    """
    Create a new notebook execution task.

    Args:
        task_name (str): Unique identifier for the task.
        notebook_path (str): Path to the Jupyter notebook (relative to workspace_dir).
        dependencies (List[str], optional): List of task IDs this task depends on.

    Returns:
        Dict[str, Any]: Response from the API.
    """
    url = f"http://localhost:5000/create_task/"
    payload = {
        "task_name": task_name,
        "notebook_path": notebook_path,
        "dependencies": dependencies or []
    }
    response = requests.post(url, json=payload)
    return response.json()

def execute_notebook(task_id: str, priority: int = 1) -> Dict[str, Any]:
    """
    Execute an existing notebook task by its ID with a given priority.

    Args:
        task_id (str): ID of the task to execute.
        priority (int, optional): Priority level (1=High, 2=Medium, 3=Low). Defaults to 1.

    Returns:
        Dict[str, Any]: Response from the API.
    """
    url = f"http://localhost:5000/execute_notebook/"
    payload = {
        "task_id": task_id,
        "priority": priority
    }
    response = requests.post(url, json=payload)
    return response.json()

def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status and metadata of a specific task.

    Args:
        task_id (str): ID of the task to query.

    Returns:
        Dict[str, Any]: Task status and metadata.
    """
    url = f"http://localhost:5000/task_status/{task_id}"
    response = requests.get(url)
    return response.json()

def create_notebook_api(notebook_id: str) -> Dict[str, Any]:
    """
    Create a new notebook via the api
    """
    url = f"http://localhost:5000/create_notebook/"
    payload = {"notebook_id": notebook_id}
    response = requests.post(url, json=payload)
    return response.json()

def insert_cell_api(notebook_id: str, cell_type: str, cell_content: str, position: Optional[int] = None) -> Dict[str, Any]:
    """
    Insert a new cell into a notebook using the API.

    Args:
        notebook_id (str): ID of the notebook to modify.
        cell_type (str): Type of the cell ('code' or 'markdown').
        cell_content (str): Content of the cell.
        position (int, optional): Position to insert the cell. Defaults to None (end).

    Returns:
        Dict[str, Any]: Response from the API.
    """
    url = f"http://localhost:5000/insert_cell/"
    payload = {
       "notebook_id": notebook_id,
       "cell_content": {"type": cell_type, "content": cell_content},
       "position": position
    }
    response = requests.post(url, json=payload)
    return response.json()


def update_cell_api(notebook_id: str, cell_index: int, cell_type: str, cell_content: str) -> Dict[str, Any]:
    """
    Update a cell content in a notebook using the API.

    Args:
        notebook_id (str): ID of the notebook to modify.
        cell_index (int): Index of the cell to update.
         cell_type (str): Type of the cell ('code' or 'markdown').
        cell_content (str): New content of the cell.

    Returns:
        Dict[str, Any]: Response from the API.
    """
    url = f"http://localhost:5000/update_cell/"
    payload = {
        "notebook_id": notebook_id,
        "cell_index": cell_index,
        "cell_content": {"type": cell_type, "content": cell_content}
    }
    response = requests.post(url, json=payload)
    return response.json()


def delete_cell_api(notebook_id: str, cell_index: int) -> Dict[str, Any]:
    """
   Delete a cell from a notebook via the API.

    Args:
        notebook_id (str): ID of the notebook to modify.
        cell_index (int): Index of the cell to delete.

    Returns:
        Dict[str, Any]: Response from the API.
   """
    url = f"http://localhost:5000/delete_cell/"
    payload = {
        "notebook_id": notebook_id,
        "cell_index": cell_index
    }
    response = requests.post(url, json=payload)
    return response.json()


def execute_cell_api(notebook_id: str, cell_index: int) -> Dict[str, Any]:
    """
   Execute a specific cell in a notebook using the API.

    Args:
        notebook_id (str): ID of the notebook containing the cell.
        cell_index (int): Index of the cell to execute.

    Returns:
        Dict[str, Any]: Response from the API.
    """
    url = f"http://localhost:5000/execute_cell/"
    payload = {
        "notebook_id": notebook_id,
        "cell_index": cell_index
    }
    response = requests.post(url, json=payload)
    return response.json()

def save_notebook_api(notebook_id: str, filepath: str) -> Dict[str, Any]:
    """
    Save a notebook to the specified filepath using the API.

    Args:
        notebook_id (str): ID of the notebook to save.
        filepath (str): Filesystem path where the notebook should be saved.

    Returns:
        Dict[str, Any]: Response from the API.
    """
    url = f"http://localhost:5000/save_notebook/"
    payload = {
        "notebook_id": notebook_id,
        "filepath": filepath
    }
    response = requests.post(url, json=payload)
    return response.json()


# -------------------- Running the Flask Server --------------------

def run_flask():
    socketio.run(app_flask, host="0.0.0.0", port=5000)

# Start Flask in a separate thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# -------------------- Persistence and Scalability --------------------

# The system uses JSON files for state persistence.
# For scalability, consider integrating a database like SQLite or PostgreSQL.

# -------------------- Conclusion --------------------

print("outERRA Framework is up and running. Use the GUI controls or Python functions to interact with the system.")
