from service.cache import TTLCache


def test_cache_expires_entries():
    now = [0.0]
    cache = TTLCache(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    cache.set("a", "value")
    assert cache.get("a") == "value"

    now[0] = 10.0
    assert cache.get("a") is None


def test_cache_evicts_least_recently_used_entry():
    cache = TTLCache(max_entries=2, ttl_seconds=10)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3
