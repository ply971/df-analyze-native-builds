; Inno Setup script for the df-analyze desktop GUI.
;
; Wraps the PyInstaller onedir build (dist\df-analyze-desktop\) into a single
; setup.exe: a normal double-click install wizard with a Start Menu shortcut
; and an uninstaller. df-analyze's full ML pipeline (torch, sklearn, catboost,
; lightgbm, ...) is bundled inside, so nothing else needs to be installed for
; the app to run.
;
; Build the PyInstaller bundle first (see desktop/README.md), then compile
; this script from the repo root:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" desktop\installer.iss
;
; The finished installer is written to dist\installer\df-analyze-desktop-setup.exe.

#define MyAppName "df-analyze Desktop"
#define MyAppVersion "4.1.0"
#define MyAppExeName "df-analyze-desktop.exe"
#ifndef MyAppSourceDir
  #define MyAppSourceDir "..\dist\df-analyze-desktop"
#endif

[Setup]
AppId={{B6C1E9C2-6E6C-4B44-9D8B-6B7F0B7E1A11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=stfxecutables
DefaultDirName={localappdata}\Programs\df-analyze-desktop
DefaultGroupName=df-analyze
DisableProgramGroupPage=yes
; Per-user install under %LOCALAPPDATA% -- no admin rights required, so
; anyone can just download and run the installer.
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=df-analyze-desktop-setup
Compression=lzma2
SolidCompression=yes
; The bundled ML stack is large (~1.5GB+); allow the biggest LZMA dictionary
; for the best compression ratio on a build this size.
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
