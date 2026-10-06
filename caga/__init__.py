"""CAGA: per-item contrastive attribution and activation steering of MLP neurons in an audio LLM.

Modules
    paths        environment and storage locations
    config       every experiment setting
    data         benchmark download -> unified manifests
    mcq          prompt formatting and answer parsing
    model        Qwen2.5-Omni Thinker loading and generation
    contrast     clean / Gaussian-noise contrast pair
    attribution  grad x act salience and selection scores
    selection    top-K neuron selection per mode -> steering specs
    steering     forward hooks that inject alpha * delta during prefill
    experiment   run_steering() and evaluate()
    results      result tables and Excel export
    selftest     GPU checks for the speed options
"""
__version__ = "30.2"
