#!/bin/bash
# Builds MiniFlow.app, installs it, and starts it at every login.
# Run once from the miniflow folder:  bash make_app.sh
set -e
cd "$(dirname "$0")"
SRC="$(pwd)"
NAME="MiniFlow"

if [ ! -x "$SRC/.venv/bin/python" ]; then
  echo "Run ./setup.sh first (the .venv folder is missing)."
  exit 1
fi

# macOS won't let a new app read Desktop, Documents or Downloads,
# so MiniFlow gets its own folder in your home folder instead.
case "$SRC" in
  "$HOME/Desktop"*|"$HOME/Documents"*|"$HOME/Downloads"*)
    NEW="$HOME/MiniFlow"
    if [ -e "$NEW" ]; then
      echo "$NEW already exists - rename or delete it, then run this again."
      exit 1
    fi
    pkill -f "miniflow.py" 2>/dev/null || true
    sleep 1
    echo "Moving MiniFlow to $NEW (so the app is allowed to open it)..."
    mv "$SRC" "$NEW"
    SRC="$NEW"
    cd "$SRC"
    ;;
esac

# Install to /Applications if we can, otherwise ~/Applications
DEST_DIR="/Applications"
[ -w "$DEST_DIR" ] || { DEST_DIR="$HOME/Applications"; mkdir -p "$DEST_DIR"; }
APP="$DEST_DIR/$NAME.app"

echo "Stopping any MiniFlow that is already running..."
pkill -f "miniflow.py" 2>/dev/null || true
sleep 1

echo "Building $APP ..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# --- the app's launcher: runs your MiniFlow code with your Python setup
cat > "$APP/Contents/MacOS/$NAME" <<EOF
#!/bin/bash
if ! cd "$SRC" 2>/dev/null || [ ! -r "$SRC/miniflow.py" ]; then
  osascript -e 'display alert "MiniFlow cannot start" message "Its files were not found in $SRC. Run make_app.sh again."'
  exit 1
fi
exec "$SRC/.venv/bin/python" -u "$SRC/miniflow.py" >> "$SRC/miniflow.log" 2>&1
EOF
chmod +x "$APP/Contents/MacOS/$NAME"

# --- app info: menu bar only (no Dock icon), and why it needs the mic
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$NAME</string>
  <key>CFBundleDisplayName</key><string>$NAME</string>
  <key>CFBundleIdentifier</key><string>in.sarthak-gupta.miniflow</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>$NAME</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSUIElement</key><true/>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSMicrophoneUsageDescription</key><string>MiniFlow listens while you hold the dictation key, to turn your voice into text.</string>
  <key>NSAppleEventsUsageDescription</key><string>MiniFlow types the text you dictate into the app you are using.</string>
</dict>
</plist>
EOF

# --- icon
if [ -f "$SRC/app/icon_1024.png" ]; then
  ICONSET="$(mktemp -d)/AppIcon.iconset"
  mkdir -p "$ICONSET"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$SRC/app/icon_1024.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2))
    sips -z $d $d "$SRC/app/icon_1024.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
fi

# Sign it locally so macOS remembers its permissions between launches
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true

echo "Adding MiniFlow to Login Items (macOS may ask to allow this)..."
osascript -e "tell application \"System Events\" to delete (every login item whose name is \"$NAME\")" >/dev/null 2>&1 || true
osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$APP\", hidden:true}" >/dev/null 2>&1 \
  && echo "  Added." || echo "  Couldn't add it - add MiniFlow yourself in System Settings > General > Login Items."

echo "Starting MiniFlow..."
open "$APP"

echo ""
echo "Done! MiniFlow is now an app in $DEST_DIR and starts when you log in."
echo ""
echo "One-time permissions (macOS treats the app as new):"
echo "  1. Click Allow when it asks for the Microphone."
echo "  2. System Settings > Privacy & Security > Accessibility     -> turn on MiniFlow"
echo "  3. System Settings > Privacy & Security > Input Monitoring  -> turn on MiniFlow"
echo "     (use + to add it from $DEST_DIR if it isn't listed)"
echo "  4. Then click the menu bar icon > Settings > Quit MiniFlow, and open MiniFlow again."
echo ""
echo "MiniFlow's files now live in: $SRC"
echo "You can close this Terminal window now."
