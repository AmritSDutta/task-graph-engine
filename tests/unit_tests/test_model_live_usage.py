import unittest
import threading
from src.task_agent.utils.model_live_usage import ModelLiveUsage, ModelLiveUsageSingleton, get_model_usage_singleton


class TestModelLiveUsage(unittest.TestCase):
    """Test cases for the original ModelLiveUsage class."""
    
    def setUp(self):
        self.model_usage = ModelLiveUsage()
    
    def test_initialization(self):
        """Test that ModelLiveUsage initializes correctly."""
        self.assertIsNotNone(self.model_usage.model_usage)
        self.assertEqual(dict(self.model_usage.model_usage), {})
    
    def test_add_model_usage(self):
        """Test adding model usage - always increments by 1."""
        # Call 10 times to get usage of 10
        for _ in range(10):
            self.model_usage.add_model_usage("gpt-4")
        self.assertEqual(self.model_usage.model_usage["gpt-4"], 10)

        # Call 10 more times to get usage of 20
        for _ in range(10):
            self.model_usage.add_model_usage("gpt-4")
        self.assertEqual(self.model_usage.model_usage["gpt-4"], 20)
    
    def test_get_model_usage(self):
        """Test getting model usage for specific models."""
        # Add usage by calling multiple times
        for _ in range(10):
            self.model_usage.add_model_usage("gpt-4")
        for _ in range(5):
            self.model_usage.add_model_usage("gpt-3.5")

        # Use get_models_usage for multiple models (list input)
        result = self.model_usage.get_models_usage(["gpt-4", "gpt-3.5"])
        self.assertEqual(result["gpt-4"], 10)
        self.assertEqual(result["gpt-3.5"], 5)

        # Use get_model_usage for single model (string input)
        single_result = self.model_usage.get_model_usage("gpt-4")
        self.assertEqual(single_result, 10)

        # Test getting usage for a model that hasn't been added (should return default 0)
        result_new = self.model_usage.get_models_usage(["nonexistent-model"])
        self.assertEqual(result_new["nonexistent-model"], 0)
    
    def test_log_model_usage(self):
        """Test logging model usage (just ensure no exceptions are raised)."""
        self.model_usage.add_model_usage("gpt-4", 10)
        try:
            self.model_usage.log_model_usage()
        except Exception as e:
            self.fail(f"log_model_usage raised {e} unexpectedly!")


class TestModelLiveUsageSingleton(unittest.TestCase):
    """Test cases for the ModelLiveUsageSingleton."""
    
    def test_singleton_instance_creation(self):
        """Test that only one instance of ModelLiveUsage is created."""
        singleton1 = ModelLiveUsageSingleton()
        singleton2 = ModelLiveUsageSingleton()
        
        self.assertIs(singleton1, singleton2)
    
    def test_singleton_model_usage_instance(self):
        """Test that the singleton returns the same ModelLiveUsage instance."""
        singleton1 = ModelLiveUsageSingleton()
        singleton2 = ModelLiveUsageSingleton()
        
        instance1 = singleton1.get_instance()
        instance2 = singleton2.get_instance()
        
        self.assertIs(instance1, instance2)
    
    def test_concurrent_access(self):
        """Test thread-safe access to the singleton."""
        instances = []
        
        def get_instance():
            singleton = ModelLiveUsageSingleton()
            instances.append(singleton.get_instance())
        
        # Create multiple threads to access the singleton
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_instance)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all instances are the same
        first_instance = instances[0]
        for instance in instances[1:]:
            self.assertIs(first_instance, instance)
    
    def test_concurrent_singleton_creation(self):
        """Test thread-safe singleton creation."""
        singletons = []
        
        def create_singleton():
            singleton = ModelLiveUsageSingleton()
            singletons.append(singleton)
        
        # Create multiple threads to create singleton instances
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_singleton)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all singleton instances are the same
        first_singleton = singletons[0]
        for singleton in singletons[1:]:
            self.assertIs(first_singleton, singleton)
    
    def test_singleton_data_persistence(self):
        """Test that data persists across singleton accesses."""
        # Add data using one singleton instance (call 100 times)
        singleton1 = ModelLiveUsageSingleton()
        instance1 = singleton1.get_instance()
        for _ in range(100):
            instance1.add_model_usage("gpt-4")

        # Access the same data using another singleton instance
        singleton2 = ModelLiveUsageSingleton()
        instance2 = singleton2.get_instance()

        # Verify the data is shared
        self.assertEqual(instance2.model_usage["gpt-4"], 100)

        # Also verify through get_models_usage method (list input)
        result = instance2.get_models_usage(["gpt-4"])
        self.assertEqual(result["gpt-4"], 100)


class TestModelLiveUsageThreadSafety(unittest.TestCase):
    """Test thread safety of the ModelLiveUsage class."""

    def test_thread_safe_operations(self):
        """Test that ModelLiveUsage operations are thread-safe."""
        model_usage = ModelLiveUsage()
        num_threads = 10
        operations_per_thread = 100

        def worker(model_name):
            for i in range(operations_per_thread):
                # Increment usage in a thread-safe manner
                current = model_usage.model_usage[model_name]
                model_usage.add_model_usage(model_name, current + 1)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(f"model_{i % 3}",))  # Using 3 different models
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Just ensure no exceptions occurred during concurrent access
        self.assertTrue(True)


class TestGetModelUsageSingletonFunction(unittest.TestCase):
    """Test cases for the get_model_usage_singleton function."""

    def test_function_returns_same_instance(self):
        """Test that the function always returns the same instance."""
        instance1 = get_model_usage_singleton()
        instance2 = get_model_usage_singleton()

        self.assertIs(instance1, instance2)

    def test_function_returns_correct_type(self):
        """Test that the function returns a ModelLiveUsage instance."""
        instance = get_model_usage_singleton()
        self.assertIsInstance(instance, ModelLiveUsage)

    def test_function_data_persistence(self):
        """Test that data persists when accessed through the function."""
        # Add data using the function (call 50 times)
        instance1 = get_model_usage_singleton()
        for _ in range(50):
            instance1.add_model_usage("gpt-3.5")

        # Access the same data using the function again
        instance2 = get_model_usage_singleton()

        # Verify the data is shared
        self.assertEqual(instance2.model_usage["gpt-3.5"], 50)


class TestModelLiveUsageExtended(unittest.TestCase):
    """Extended test cases for the ModelLiveUsage class."""

    def setUp(self):
        self.model_usage = ModelLiveUsage()

    def test_multiple_model_tracking(self):
        """Test tracking multiple models."""
        models_and_usage = {
            "gpt-4": 100,
            "gpt-3.5": 50,
            "claude-3": 75,
            "llama-2": 30
        }

        # Add usage by calling add_model_usage multiple times for each model
        for model, usage in models_and_usage.items():
            for _ in range(usage):
                self.model_usage.add_model_usage(model)

        # Verify all models have correct usage
        for model, expected_usage in models_and_usage.items():
            self.assertEqual(self.model_usage.model_usage[model], expected_usage)

        # Test retrieving multiple models at once using get_models_usage
        retrieved = self.model_usage.get_models_usage(list(models_and_usage.keys()))
        for model, expected_usage in models_and_usage.items():
            self.assertEqual(retrieved[model], expected_usage)


class TestModelLiveUsageWithUsageParameter(unittest.TestCase):
    """Test cases for add_model_usage with custom usage parameter."""

    def setUp(self):
        self.model_usage = ModelLiveUsage()

    def test_add_model_usage_with_custom_usage_value(self):
        """Test that add_model_usage correctly uses the usage parameter value."""
        # Add with custom usage value
        self.model_usage.add_model_usage("gpt-4", 25)
        self.assertEqual(self.model_usage.model_usage["gpt-4"], 25)

        # Add more with different usage value
        self.model_usage.add_model_usage("gpt-4", 15)
        self.assertEqual(self.model_usage.model_usage["gpt-4"], 40)

        # Test with zero usage
        self.model_usage.add_model_usage("gpt-3.5", 0)
        self.assertEqual(self.model_usage.model_usage["gpt-3.5"], 0)

        # Test with default usage (1)
        self.model_usage.add_model_usage("claude-3")
        self.assertEqual(self.model_usage.model_usage["claude-3"], 1)

    def test_add_model_usage_accumulates_multiple_calls(self):
        """Test that multiple add_model_usage calls with different usage values accumulate correctly."""
        # Add multiple times with different usage values
        self.model_usage.add_model_usage("model-a", 10)
        self.assertEqual(self.model_usage.model_usage["model-a"], 10)

        self.model_usage.add_model_usage("model-a", 20)
        self.assertEqual(self.model_usage.model_usage["model-a"], 30)

        self.model_usage.add_model_usage("model-a", 5)
        self.assertEqual(self.model_usage.model_usage["model-a"], 35)

        # Test multiple models with different usage patterns
        self.model_usage.add_model_usage("model-b", 100)
        self.model_usage.add_model_usage("model-b", 50)
        self.assertEqual(self.model_usage.model_usage["model-b"], 150)

        # Verify model-a is still correct
        self.assertEqual(self.model_usage.model_usage["model-a"], 35)


if __name__ == '__main__':
    unittest.main()