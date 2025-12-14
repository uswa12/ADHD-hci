# core_intelligence/task_manager.py
# Dynamic, logic-based task chunking (ADHD-friendly)

import math

class TaskManager:
    """
    Dynamically breaks user tasks into focused chunks
    using rule-based cognitive logic (AI-like behavior).
    """

    def __init__(self):
        self.user_task_title = ""
        self.chunked_tasks = []
        self.task_pointer = 0

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def set_task(self, title: str):
        """
        Takes a user task and dynamically breaks it down.
        """
        self.user_task_title = title
        self.chunked_tasks = []
        self.task_pointer = 0

        task_type = self._detect_task_type(title)
        complexity = self._estimate_complexity(title)
        block_duration = self._recommend_block_duration(complexity)
        total_blocks = self._estimate_number_of_blocks(complexity)

        for i in range(total_blocks):
            self.chunked_tasks.append({
                "id": i + 1,
                "title": self._generate_chunk_title(task_type, i + 1),
                "duration": block_duration
            })

        return self.chunked_tasks

    def get_current_task_info(self):
        if not self.chunked_tasks:
            return "General Study Session"

        task = self.chunked_tasks[self.task_pointer]
        return f"Task {task['id']}: {task['title']} ({task['duration']} min)"

    # --------------------------------------------------
    # CORE INTELLIGENCE LOGIC
    # --------------------------------------------------

    def _detect_task_type(self, title):
        title = title.lower()

        if any(k in title for k in ["essay", "report", "assignment", "write"]):
            return "writing"
        if any(k in title for k in ["code", "program", "debug", "project"]):
            return "coding"
        if any(k in title for k in ["study", "read", "chapter", "notes"]):
            return "reading"
        if any(k in title for k in ["revise", "review", "practice"]):
            return "revision"

        return "general"

    def _estimate_complexity(self, title):
        """
        Complexity score (1–10)
        """
        score = 1
        length_factor = len(title.split())

        score += min(length_factor // 3, 3)

        keywords = [
            "analysis", "design", "research", "final",
            "implementation", "optimization", "case study"
        ]

        for word in keywords:
            if word in title.lower():
                score += 2

        return min(score, 10)

    def _recommend_block_duration(self, complexity):
        """
        ADHD-optimized block length
        """
        if complexity <= 3:
            return 10
        elif complexity <= 6:
            return 15
        elif complexity <= 8:
            return 20
        else:
            return 25

    def _estimate_number_of_blocks(self, complexity):
        """
        More complex tasks → more chunks
        """
        return max(2, math.ceil(complexity / 2))

    def _generate_chunk_title(self, task_type, index):
        templates = {
            "writing": [
                "Brainstorm & Outline",
                "Draft Core Content",
                "Expand Arguments",
                "Refine Structure",
                "Proofread & Polish"
            ],
            "coding": [
                "Understand Requirements",
                "Implement Core Logic",
                "Add Features",
                "Debug & Optimize",
                "Refactor & Clean Code"
            ],
            "reading": [
                "Preview Key Topics",
                "Focused Reading",
                "Take Notes",
                "Summarize Concepts"
            ],
            "revision": [
                "Recall Key Concepts",
                "Practice Problems",
                "Identify Weak Areas",
                "Final Review"
            ],
            "general": [
                "Initial Focus Block",
                "Deep Work Session",
                "Consolidation Phase"
            ]
        }

        options = templates.get(task_type, templates["general"])

        if index <= len(options):
            return options[index - 1]
        return f"Focus Block {index}"  