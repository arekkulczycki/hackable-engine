*** Troubleshooting

To avoid warnings about shared_memory the following patch was done on multiprocessing shared_memory.py

```
if create:
    from .resource_tracker import register
    register(self._name, "shared_memory")
```