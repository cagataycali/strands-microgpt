"""Test that all imports work without heavy dependencies."""


def test_value_import():
    from strands_microgpt import Value

    assert Value is not None


def test_microgpt_import():
    from strands_microgpt import MicroGPT

    assert MicroGPT is not None


def test_tokenizer_import():
    from strands_microgpt import Tokenizer

    assert Tokenizer is not None


def test_model_import():
    from strands_microgpt import MicroGPTModel

    assert MicroGPTModel is not None


def test_tools_import():
    from strands_microgpt import microgpt_generate, microgpt_train

    assert microgpt_train is not None
    assert microgpt_generate is not None
