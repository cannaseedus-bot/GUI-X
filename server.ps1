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

# ── Main Request Dispatcher ───────────────────────────────────
function Dispatch($req, $res) {
    $url    = $req.Url.AbsolutePath.TrimEnd('/')
    $method = $req.HttpMethod.ToUpper()

    Write-Log "$method $($req.Url.PathAndQuery)"

    switch -Regex ($url) {
        '^/players$'     { Handle-Players  $req $res; return }
        '^/presence$'    { Handle-Presence $req $res; return }
        '^/chat_global$' { Handle-GlobalChat $req $res; return }
        '^/chat_dm$'     { Handle-DmChat   $req $res; return }
        '^/health$'      { Send-Json $res @{ status = 'ok'; time = (Get-Date -Format 'o') }; return }
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
