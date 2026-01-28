# Where to find pod demo?
[kubernetes pod](https://kubernetes.io/docs/concepts/workloads/pods/)
and search "apiVer"

or we can

```bash
k run nginx-yaml --image=nginx --dry-run=client -o yaml
```

# How to create pod using .yaml?
```
k create -f nginx.yaml
# or
k apply -f nginx.yaml
```
