# ============================================================
#  GUI-X  |  Glass Matrix OS  —  PowerShell HTTP Server
#  Serves static files + REST API on http://localhost:3000
# ============================================================

param(
    [int]$Port = 3000,
    [string]$Root = $PSScriptRoot
)

# ── In-memory stores ─────────────────────────────────────────
$script:players      = [System.Collections.Generic.List[hashtable]]::new()
$script:globalChat   = [System.Collections.Generic.List[hashtable]]::new()
$script:dmMessages   = [System.Collections.Generic.List[hashtable]]::new()
$script:presence     = @{}

# ── Graph inference stores ────────────────────────────────────
# SHA256 → cached output graph (JSON string)
$script:graphCache   = [System.Collections.Generic.Dictionary[string,string]]::new()
$script:graphLRU     = [System.Collections.Generic.List[string]]::new()
$script:GRAPH_CACHE_MAX = 2048

# ── MIME map ─────────────────────────────────────────────────
$MimeTypes = @{
    '.html' = 'text/html; charset=utf-8'
    '.css'  = 'text/css; charset=utf-8'
    '.js'   = 'application/javascript; charset=utf-8'
    '.json' = 'application/json; charset=utf-8'
    '.png'  = 'image/png'
    '.jpg'  = 'image/jpeg'
    '.jpeg' = 'image/jpeg'
    '.gif'  = 'image/gif'
    '.svg'  = 'image/svg+xml'
    '.ico'  = 'image/x-icon'
    '.woff' = 'font/woff'
    '.woff2'= 'font/woff2'
    '.ttf'  = 'font/ttf'
    '.zip'  = 'application/zip'
    '.txt'  = 'text/plain; charset=utf-8'
    '.wasm' = 'application/wasm'
    '.wat'  = 'text/plain; charset=utf-8'
    '.xml'  = 'application/xml; charset=utf-8'
}

# ── Helpers ──────────────────────────────────────────────────
function Write-Log($msg) {
    $ts = (Get-Date).ToString('HH:mm:ss')
    Write-Host "[$ts] $msg" -ForegroundColor Cyan
}

function Send-Json($response, $obj, [int]$status = 200) {
    $body    = $obj | ConvertTo-Json -Depth 10 -Compress
    $bytes   = [System.Text.Encoding]::UTF8.GetBytes($body)
    $response.StatusCode        = $status
    $response.ContentType       = 'application/json; charset=utf-8'
    $response.ContentLength64   = $bytes.Length
    $response.Headers.Add('Access-Control-Allow-Origin', '*')
    $response.Headers.Add('Access-Control-Allow-Methods', 'GET, POST, PATCH, OPTIONS')
    $response.Headers.Add('Access-Control-Allow-Headers', 'Content-Type')
    $response.OutputStream.Write($bytes, 0, $bytes.Length)
    $response.OutputStream.Close()
}

function Read-Body($request) {
    $reader = [System.IO.StreamReader]::new($request.InputStream, $request.ContentEncoding)
    $raw    = $reader.ReadToEnd()
    if ($raw -and $raw.Trim()) {
        return $raw | ConvertFrom-Json -AsHashtable
    }
    return @{}
}

function Serve-Static($response, $path) {
    if (-not (Test-Path $path -PathType Leaf)) {
        $response.StatusCode = 404
        $response.Close()
        return
    }
    $ext   = [System.IO.Path]::GetExtension($path).ToLower()
    $mime  = if ($MimeTypes.ContainsKey($ext)) { $MimeTypes[$ext] } else { 'application/octet-stream' }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    $response.StatusCode        = 200
    $response.ContentType       = $mime
    $response.ContentLength64   = $bytes.Length
    $response.Headers.Add('Access-Control-Allow-Origin', '*')
    $response.OutputStream.Write($bytes, 0, $bytes.Length)
    $response.OutputStream.Close()
}

# ── API Route Handlers ────────────────────────────────────────

function Handle-Players($req, $res) {
    $method = $req.HttpMethod.ToUpper()

    if ($method -eq 'OPTIONS') { Send-Json $res @{}; return }

    if ($method -eq 'GET') {
        # Prune stale players (no heartbeat for > 60s)
        $cutoff = (Get-Date).AddSeconds(-60)
        $script:players = [System.Collections.Generic.List[hashtable]]($script:players | Where-Object {
            $pid = $_['id']
            $last = $script:presence[$pid]
            $last -and $last -gt $cutoff
        })
        Send-Json $res ($script:players | ForEach-Object { $_ })
        return
    }

    if ($method -eq 'POST') {
        $body = Read-Body $req
        $existing = $script:players | Where-Object { $_['id'] -eq $body['id'] }
        if (-not $existing) {
            $player = @{
                id       = $body['id']
                name     = $body['name']
                avatar   = $body['avatar']
                joinedAt = (Get-Date -Format 'o')
            }
            $script:players.Add($player)
            $script:presence[$body['id']] = Get-Date
        }
        Send-Json $res @{ ok = $true }
        return
    }

    $res.StatusCode = 405; $res.Close()
}

function Handle-Presence($req, $res) {
    $method = $req.HttpMethod.ToUpper()
    if ($method -eq 'OPTIONS') { Send-Json $res @{}; return }
    if ($method -eq 'PATCH') {
        $body = Read-Body $req
        if ($body['id']) {
            $script:presence[$body['id']] = Get-Date
        }
        Send-Json $res @{ ok = $true }
        return
    }
    $res.StatusCode = 405; $res.Close()
}

function Handle-GlobalChat($req, $res) {
    $method = $req.HttpMethod.ToUpper()
    if ($method -eq 'OPTIONS') { Send-Json $res @{}; return }

    if ($method -eq 'GET') {
        $since = $req.QueryString['since']
        $msgs = if ($since) {
            $dt = [datetime]::Parse($since)
            $script:globalChat | Where-Object { [datetime]::Parse($_['ts']) -gt $dt }
        } else {
            $script:globalChat | Select-Object -Last 50
        }
        Send-Json $res ($msgs | ForEach-Object { $_ })
        return
    }

    if ($method -eq 'POST') {
        $body = Read-Body $req
        $msg = @{
            id     = [System.Guid]::NewGuid().ToString()
            from   = $body['from']
            name   = $body['name']
            avatar = $body['avatar']
            text   = $body['text']
            ts     = (Get-Date -Format 'o')
        }
        $script:globalChat.Add($msg)
        # Keep last 200 messages
        if ($script:globalChat.Count -gt 200) {
            $script:globalChat.RemoveAt(0)
        }
        Send-Json $res @{ ok = $true; id = $msg['id'] }
        return
    }

    $res.StatusCode = 405; $res.Close()
}

function Handle-DmChat($req, $res) {
    $method = $req.HttpMethod.ToUpper()
    if ($method -eq 'OPTIONS') { Send-Json $res @{}; return }

    if ($method -eq 'GET') {
        $a    = $req.QueryString['a']
        $b    = $req.QueryString['b']
        $since = $req.QueryString['since']
        $msgs = $script:dmMessages | Where-Object {
            ($_ ['from'] -eq $a -and $_['to'] -eq $b) -or
            ($_ ['from'] -eq $b -and $_['to'] -eq $a)
        }
        if ($since) {
            $dt   = [datetime]::Parse($since)
            $msgs = $msgs | Where-Object { [datetime]::Parse($_['ts']) -gt $dt }
        }
        Send-Json $res ($msgs | ForEach-Object { $_ })
        return
    }

    if ($method -eq 'POST') {
        $body = Read-Body $req
        $msg = @{
            id     = [System.Guid]::NewGuid().ToString()
            from   = $body['from']
            to     = $body['to']
            name   = $body['name']
            avatar = $body['avatar']
            text   = $body['text']
            ts     = (Get-Date -Format 'o')
        }
        $script:dmMessages.Add($msg)
        Send-Json $res @{ ok = $true; id = $msg['id'] }
        return
    }

    $res.StatusCode = 405; $res.Close()
}

# ── Graph: SHA256 helper ──────────────────────────────────────
function Compute-Sha256([string]$str) {
    $sha   = [System.Security.Cryptography.SHA256]::Create()
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($str)
    $hash  = $sha.ComputeHash($bytes)
    $sha.Dispose()
    return ([System.BitConverter]::ToString($hash) -replace '-','').ToLower()
}

# ── Graph: JS-level 8D inference (pure PS, deterministic) ─────
# Mirrors tensor_graph.js stepJS() logic server-side.
function Step-Graph8D([hashtable]$graphObj) {
    $nodes    = $graphObj['clusters']
    $edges    = @($graphObj['edges'])
    $step     = [int]($graphObj['step'] ?? 0)
    $epsilon  = 1e-5
    $forceMax = 1e4

    # Flatten all nodes into id→node map
    $nodeMap = @{}
    foreach ($clusterKey in $nodes.Keys) {
        foreach ($n in $nodes[$clusterKey]) {
            $nodeMap[[string]$n['id']] = $n
        }
    }

    foreach ($nodeId in $nodeMap.Keys) {
        $node = $nodeMap[$nodeId]
        $ctx  = @(0.0, 0.0, 0.0)

        # Accumulate weighted context from incoming attention edges
        foreach ($e in $edges) {
            if ([string]$e['to'] -eq $nodeId -and $e['type'] -eq 'attention') {
                $fromNode = $nodeMap[[string]$e['from']]
                if ($fromNode) {
                    $w = [float]$e['weight']
                    $ctx[0] += $w * [float]$fromNode['px']
                    $ctx[1] += $w * [float]$fromNode['py']
                    $ctx[2] += $w * [float]$fromNode['pz']
                }
            }
        }

        # Force = context - self_pos, clamped
        $node['fx'] = [math]::Max(-$forceMax, [math]::Min($forceMax, $ctx[0] - [float]$node['px']))
        $node['fy'] = [math]::Max(-$forceMax, [math]::Min($forceMax, $ctx[1] - [float]$node['py']))
        $node['fz'] = [math]::Max(-$forceMax, [math]::Min($forceMax, $ctx[2] - [float]$node['pz']))

        # Signal = best gate score index (argmax of d0..d7)
        $bestD = -1e9; $bestK = 0
        for ($k = 0; $k -lt 8; $k++) {
            $dv = [float]($node["d$k"] ?? 0)
            if ($dv -gt $bestD) { $bestD = $dv; $bestK = $k }
        }
        $node['signal'] = $bestK
    }

    # Rebuild clusters by expert (argmax)
    $newClusters = @{}
    for ($e = 0; $e -lt 8; $e++) { $newClusters["$e"] = @() }
    foreach ($n in $nodeMap.Values) {
        $exp = [int]($n['signal'] ?? 0) % 8
        $newClusters["$exp"] += $n
    }

    $graphObj['clusters'] = $newClusters
    $graphObj['step']     = $step + 1
    return $graphObj
}

# ── Graph handlers ────────────────────────────────────────────

function Handle-GraphInfer($req, $res) {
    if ($req.HttpMethod.ToUpper() -eq 'OPTIONS') { Send-Json $res @{}; return }
    if ($req.HttpMethod.ToUpper() -ne 'POST')    { $res.StatusCode = 405; $res.Close(); return }

    $body = Read-Body $req
    if (-not $body) { Send-Json $res @{ error = 'empty body' } 400; return }

    # Check cache by input hash
    $canonical = ($body | ConvertTo-Json -Depth 15 -Compress)
    $inputHash = Compute-Sha256 $canonical

    if ($script:graphCache.ContainsKey($inputHash)) {
        $cached = $script:graphCache[$inputHash] | ConvertFrom-Json -AsHashtable
        $cached['_fromCache'] = $true
        Send-Json $res $cached
        return
    }

    # Run inference
    $result    = Step-Graph8D $body
    $outJson   = $result | ConvertTo-Json -Depth 15 -Compress
    $outHash   = Compute-Sha256 $outJson
    $result['sha256'] = $outHash

    # Store in LRU cache
    if ($script:graphCache.Count -ge $script:GRAPH_CACHE_MAX) {
        $oldest = $script:graphLRU[0]
        $script:graphLRU.RemoveAt(0)
        $script:graphCache.Remove($oldest)
    }
    $script:graphCache[$inputHash] = ($result | ConvertTo-Json -Depth 15 -Compress)
    $script:graphLRU.Add($inputHash)

    Send-Json $res $result
}

function Handle-GraphHash($req, $res) {
    if ($req.HttpMethod.ToUpper() -eq 'OPTIONS') { Send-Json $res @{}; return }
    if ($req.HttpMethod.ToUpper() -ne 'POST')    { $res.StatusCode = 405; $res.Close(); return }

    $body      = Read-Body $req
    $canonical = ($body | ConvertTo-Json -Depth 15 -Compress)
    $hash      = Compute-Sha256 $canonical
    Send-Json $res @{ sha256 = $hash }
}

function Handle-GraphFetch($req, $res, [string]$hash) {
    if ($script:graphCache.ContainsKey($hash)) {
        $cached = $script:graphCache[$hash] | ConvertFrom-Json -AsHashtable
        Send-Json $res $cached
    } else {
        Send-Json $res @{ error = 'not found' } 404
    }
}

# ── Main Request Dispatcher ───────────────────────────────────
function Dispatch($req, $res) {
    $url    = $req.Url.AbsolutePath.TrimEnd('/')
    $method = $req.HttpMethod.ToUpper()

    Write-Log "$method $($req.Url.PathAndQuery)"

    switch -Regex ($url) {
        '^/players$'          { Handle-Players    $req $res; return }
        '^/presence$'         { Handle-Presence   $req $res; return }
        '^/chat_global$'      { Handle-GlobalChat $req $res; return }
        '^/chat_dm$'          { Handle-DmChat     $req $res; return }
        '^/graph/infer$'      { Handle-GraphInfer $req $res; return }
        '^/graph/hash$'       { Handle-GraphHash  $req $res; return }
        '^/graph/([0-9a-f]+)$' {
            $hash = [regex]::Match($url, '^/graph/([0-9a-f]+)$').Groups[1].Value
            Handle-GraphFetch $req $res $hash; return
        }
        '^/health$'           {
            Send-Json $res @{
                status      = 'ok'
                time        = (Get-Date -Format 'o')
                graph_cache = $script:graphCache.Count
            }
            return
        }
    }

    # Static file serving
    $filePath = if ($url -eq '' -or $url -eq '/') {
        Join-Path $Root 'index.html'
    } else {
        Join-Path $Root ($url.TrimStart('/').Replace('/', [System.IO.Path]::DirectorySeparatorChar))
    }

    Serve-Static $res $filePath
}

# ── Start Listener ────────────────────────────────────────────
$listener = [System.Net.HttpListener]::new()
$prefix   = "http://+:$Port/"
$listener.Prefixes.Add($prefix)

try {
    $listener.Start()
} catch {
    # Fallback to localhost-only if elevated rights not available
    $listener = [System.Net.HttpListener]::new()
    $listener.Prefixes.Add("http://localhost:$Port/")
    $listener.Start()
}

Write-Host ""
Write-Host "  ╔════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║   GUI-X  |  Glass Matrix OS  v1.0      ║" -ForegroundColor Green
Write-Host "  ║   Server running on port $Port           ║" -ForegroundColor Green
Write-Host "  ║   http://localhost:$Port                ║" -ForegroundColor Green
Write-Host "  ║   Press Ctrl+C to stop                 ║" -ForegroundColor Green
Write-Host "  ╚════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# Save PID for stop script
$PidFile = Join-Path $PSScriptRoot '.server.pid'
$PID | Out-File $PidFile -Encoding ascii

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        # Run each request in a runspace to stay non-blocking
        $req = $context.Request
        $res = $context.Response
        try {
            Dispatch $req $res
        } catch {
            Write-Log "ERROR: $_"
            try { $res.StatusCode = 500; $res.Close() } catch {}
        }
    }
} finally {
    $listener.Stop()
    $listener.Close()
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
    Write-Log "Server stopped."
}
